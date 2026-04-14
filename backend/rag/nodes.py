from rag.state import RAGState
from rag.retriever import retrieve
from rag.chain import (
    route_question,
    naive_rag,
    multi_query_rag,
    contextual_compression_rag,
    _answer_stream,
    _format_docs,
    _invoke_with_retry,
    llm,
    MULTI_QUERY_PROMPT,
    COMPRESSION_PROMPT,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

STRATEGY_MAP = {
    "naive": naive_rag,
    "multi_query": multi_query_rag,
    "compression": contextual_compression_rag,
}


def _format_history(chat_history: list[dict]) -> str:
      """Convert list of dicts into a readable string for the LLM prompt."""
      if not chat_history:
          return "No previous conversation."
      lines = []
      for msg in chat_history[-10:]:  # only last 10 messages to avoid token overflow
          role = "User" if msg["role"] == "user" else "Assistant"
          lines.append(f"{role}: {msg['content']}")
      return "\n".join(lines)

def router_node(state:RAGState) -> dict: 
    """
    First node in the graph. 
    Reads the question, calls route_question() from chain.py, 
    Writes the chosen strategy back to the state. 
    """
    history_str = _format_history(state.get("chat_history", []))
    strategy = route_question(state["question"])
    return {"strategy": strategy}


def rag_node(state: RAGState) -> dict: 
    """Second node in the graph. Reads strategy from state, looks up the right function from STRATEGY_MAP, runs it, write answer back tino state."""
    strategy = state["strategy"]
    chain_fn = STRATEGY_MAP[strategy]
    history_str = _format_history(state.get("chat_history", []))
    answer = chain_fn(state["question"], state["meeting_ids"], history_str)

     # Build updated chat history with current Q&A appended
    updated_history = list(state.get("chat_history", []))
    updated_history.append({"role": "user", "content": state["question"]})
    updated_history.append({"role": "assistant", "content": answer})

    return {"answer": answer, "chat_history": updated_history}


def _retrieve_for_strategy(strategy: str, question: str, meeting_ids: list[str], chat_history: str = "") -> str:
    """
    Runs ONLY the retrieval part of each strategy, returns the context string.

    For streaming, we split the pipeline into two parts:
    1. Retrieve context (fast, non-streamable) — this function
    2. Stream the answer (slow, streamable) — _answer_stream() in chain.py

    This way the user sees tokens appearing while the LLM generates,
    instead of waiting for the entire pipeline to finish.
    """
    if strategy == "naive":
        docs = retrieve(question, meeting_ids, k=5)
        return _format_docs(docs)

    elif strategy == "multi_query":
        variants_text = _invoke_with_retry(MULTI_QUERY_PROMPT | llm | StrOutputParser(), {
            "question": question
        })
        variants = [q.strip() for q in variants_text.strip().split("\n") if q.strip()]
        seen: set[str] = set()
        all_docs: list[Document] = []
        for variant in variants:
            for doc in retrieve(variant, meeting_ids, k=3):
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        return _format_docs(all_docs[:8])

    elif strategy == "compression":
        docs = retrieve(question, meeting_ids, k=10)
        compressed = _invoke_with_retry(COMPRESSION_PROMPT | llm | StrOutputParser(), {
            "question": question,
            "context": _format_docs(docs),
        })
        return compressed

    else:
        docs = retrieve(question, meeting_ids, k=5)
        return _format_docs(docs)