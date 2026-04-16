import re
import time
import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from config.settings import settings

# Matches speaker labels like "Rhea:", "Speaker 1:", "Dr. Smith:"
SPEAKER_PATTERN = re.compile(r"\n(?=(?:[A-Z][a-zA-Z. ]+|Speaker \d+):)")

logger = logging.getLogger(__name__)


def _doc_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Embedding model for storing documents — uses retrieval_document task type."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_document",
    )


def _query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Embedding model for search queries — uses retrieval_query task type."""
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        task_type="retrieval_query",
    )


def get_vectorstore(meeting_id: str, for_query: bool = False) -> PGVector:
    """
    Return a PGVector collection scoped to a single meeting.

    Uses separate embedding objects for document storage vs query time
    to match how Gemini embedding model was trained.
    """
    embeddings = _query_embeddings() if for_query else _doc_embeddings()
    return PGVector(
        collection_name=f"meeting_{meeting_id}",
        embeddings=embeddings,
        connection=settings.database_url,
        use_jsonb=True,
    )


def _split_by_speaker(transcript: str) -> list[str]:
    """
    Split transcript on speaker labels. Each chunk is one speaker turn.
    If a turn exceeds chunk_size, sub-split it with RecursiveCharacterTextSplitter.
    Returns None-like empty list if no speaker labels found.
    """
    turns = SPEAKER_PATTERN.split(transcript)
    turns = [t.strip() for t in turns if t.strip()]

    if len(turns) <= 1:
        return []  # no speaker labels found — signal to use fallback

    # Sub-split long turns, keep short ones as-is
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
    )

    chunks = []
    for turn in turns:
        if len(turn) <= settings.chunk_size:
            chunks.append(turn)
        else:
            sub_chunks = fallback_splitter.split_text(turn)
            chunks.extend(sub_chunks)

    return chunks


def ingest_transcript(transcript: str, meeting_id: str) -> int:
    """
    Split a transcript into chunks and store them in PGVector.

    Strategy:
    1. Try speaker-aware chunking (split on "Name:" labels)
    2. If no speaker labels found, fall back to fixed-size character splitting

    Returns the number of chunks ingested.
    """
    chunks = _split_by_speaker(transcript)

    if not chunks:
        logger.info("No speaker labels found — using fixed-size chunking")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        chunks = splitter.split_text(transcript)
    else:
        logger.info(f"Speaker-aware chunking: {len(chunks)} chunks")

    docs = [
        Document(
            page_content=chunk,
            metadata={"meeting_id": meeting_id, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]

    vectorstore = get_vectorstore(meeting_id, for_query=False)
    vectorstore.add_documents(docs)
    return len(docs)


def _get_all_docs(meeting_id: str) -> list[Document]:
    """Fetch all documents from PGVector collection for BM25 using direct SQL."""
    from sqlalchemy import create_engine, text
    engine = create_engine(settings.database_url)
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT e.document, e.cmetadata
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = :collection_name
            """),
            {"collection_name": f"meeting_{meeting_id}"},
        ).fetchall()
    return [
        Document(page_content=row[0], metadata=row[1])
        for row in rows if row[0]
    ]


def _reciprocal_rank_fusion(ranked_lists: list[list[Document]], weights: list[float], rrf_k: int = 60) -> list[Document]:
    """Merge multiple ranked doc lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for docs, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(docs):
            key = doc.page_content
            doc_map[key] = doc
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [doc_map[k] for k in sorted_keys]


def retrieve(query: str, meeting_ids: list[str], k: int = 3, max_retries: int = 5) -> list[Document]:
    """
    Hybrid retrieval:
      - Semantic search (PGVector)
      - Keyword search (BM25)
      - Merged with Reciprocal Rank Fusion (weights: 0.5 / 0.5)
    """
    all_docs: list[Document] = []

    for meeting_id in meeting_ids:
        vs = get_vectorstore(meeting_id, for_query=True)
        vector_retriever = vs.as_retriever(search_kwargs={"k": k})

        collection_docs = _get_all_docs(meeting_id)
        if not collection_docs:
            for attempt in range(max_retries):
                try:
                    results = vector_retriever.invoke(query)
                    all_docs.extend(results)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait = 2 ** (attempt + 1)
                        logger.warning(f"Embedding API error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                        time.sleep(wait)
                    else:
                        raise
            continue

        bm25_retriever = BM25Retriever.from_documents(collection_docs, k=k)

        for attempt in range(max_retries):
            try:
                semantic_results = vector_retriever.invoke(query)
                bm25_results = bm25_retriever.invoke(query)
                merged = _reciprocal_rank_fusion(
                    [semantic_results, bm25_results],
                    weights=[0.5, 0.5],
                )
                all_docs.extend(merged)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Embedding API error (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                    time.sleep(wait)
                else:
                    logger.error(f"Retrieval failed after {max_retries} attempts: {e}")
                    raise

    return all_docs[:k]


def delete_meeting(meeting_id: str) -> None:
    """Remove all vectors for a meeting from PGVector."""
    vs = get_vectorstore(meeting_id)
    vs.delete_collection()
