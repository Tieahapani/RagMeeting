from typing import TypedDict, Optional


class RAGState(TypedDict):
    """
    Shared state passed between all nodes in the LangGraph RAG graph.

    question    : the user's query
    meeting_ids : which meetings to search across
    strategy    : This is set automatically by router node to select which strategy to pick depending upon the question 
    answer      : final answer string (single strategy)
    chat_history : list of previous Q&A pairs for context awareness 
    """
    question: str
    meeting_ids: list[str]
    strategy: str
    answer: str
    chat_history: list[dict]



