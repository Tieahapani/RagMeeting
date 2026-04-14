import re
import time
import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
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


def get_vectorstore(meeting_id: str, for_query: bool = False) -> Chroma:
    """
    Return a Chroma collection scoped to a single meeting.

    Uses separate embedding objects for document storage vs query time
    to match how text-embedding-004 was trained.
    """
    embeddings = _query_embeddings() if for_query else _doc_embeddings()
    return Chroma(
        collection_name=f"meeting_{meeting_id}",
        embedding_function=embeddings,
        persist_directory=settings.chroma_db_path,
        collection_metadata={"hnsw:space": "cosine"},
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
            # Long monologue — sub-split but keep speaker label on first sub-chunk
            sub_chunks = fallback_splitter.split_text(turn)
            chunks.extend(sub_chunks)

    return chunks


def ingest_transcript(transcript: str, meeting_id: str) -> int:
    """
    Split a transcript into chunks and store them in Chroma.

    Strategy:
    1. Try speaker-aware chunking (split on "Name:" labels)
    2. If no speaker labels found, fall back to fixed-size character splitting

    Returns the number of chunks ingested.
    """
    # Try speaker-aware first
    chunks = _split_by_speaker(transcript)

    # Fallback: no speaker labels detected
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

def _get_all_docs(vs: Chroma) -> list[Document]: 
    """Fetch all documents from a Chroma colletion for BM25."""
    result = vs.get()
    if not result or not result.get("documents"): 
        return []
    return [
        Document(page_content=content, metadata=meta)
        for content, meta in zip(result["documents"], result["metadatas"])
    ]

def retrieve(query: str, meeting_ids: list[str], k: int = 3, max_retries: int = 5) -> list[Document]:
    """
    Hybrid retrieval using EnsembleRetriever:                                                                                                                            
      - Semantic search (Chroma vector store)                                                                                                                                
      - Keyword search (BM25)
      - Merged with Reciprocal Rank Fusion (weights: 0.5 / 0.5)  
    """
    all_docs: list[Document] = []

    for meeting_id in meeting_ids:
        vs = get_vectorstore(meeting_id, for_query=True)

        # Semantic retriever from Chroma
        vector_retriever = vs.as_retriever(search_kwargs={"k": k})

        # BM25 retriever from same docs
        collection_docs = _get_all_docs(vs)
        if not collection_docs:
            # Fallback: semantic only if collection is empty
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

        # Combine: 50% semantic + 50% keyword
        ensemble = EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.5, 0.5],
        )

        for attempt in range(max_retries):
            try:
                results = ensemble.invoke(query)
                all_docs.extend(results)
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
    """Remove all vectors for a meeting from Chroma."""
    vs = get_vectorstore(meeting_id)
    vs.delete_collection()
