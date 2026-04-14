import time
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from config.settings import settings
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

# ── LLM provider (switchable at runtime) ─────────────────────────────────────

_current_provider = settings.llm_provider  # "gemini" or "ollama"

def _build_llm(provider: str):
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=settings.ollama_model, temperature=0)
    else:
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )

llm = _build_llm(_current_provider)

def get_provider() -> str:
    return _current_provider

def set_provider(provider: str):
    global _current_provider, llm
    _current_provider = provider
    llm = _build_llm(provider)
    logger.info(f"Switched LLM provider to: {provider}")

ROUTER_PROMPT = ChatPromptTemplate.from_template("""
You are a retrieval strategy selector for a meeting assistant.
Pick the best strategy for this question:

- naive       : specific factual question ("who attended the March 5 meeting?")
- multi_query  : broad topic question or vague question ("what did we discuss about the product roadmap?", "budget?", "what about the deadline?")
- compression  : needs a precise detailed answer from a long discussion
- reject      : not a question about a meeting — greetings, chitchat, acknowledgments, or nonsense ("okay got it", "hello", "thanks", "lol")

  Previous conversation:
  {chat_history}

Question: {question}

Reply with exactly one word — naive, multi_query, compression, or reject:""")

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a meeting assistant. Answer the question using only the meeting context below.
If the context does not contain enough information, say so clearly.
Respond in plain text only. Do not use markdown, asterisks, bullet points, or any special formatting.
Use simple sentences and line breaks to organize your answer.

   Previous conversation: 
{chat_history}
                                                 
Context:
{context}

Question: {question}

Answer:""")


MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
Generate 3 different versions of this question to improve search coverage.
Write one question per line, no numbering, no extra text.

Original: {question}

3 versions:""")

COMPRESSION_PROMPT = ChatPromptTemplate.from_template("""
From the context below extract only the sentences directly relevant to the question.
Return only those sentences, nothing else.

Question: {question}

Context:
{context}

Relevant sentences:""")



# ── Shared helpers ────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def _invoke_with_retry(chain, inputs: dict, max_retries: int = 3) -> str:
    """Invoke a LangChain chain with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning(f"LLM call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                raise


def _answer(context: str, question: str, chat_history: str = "") -> str:                                                                                                   
      return _invoke_with_retry(ANSWER_PROMPT | llm | StrOutputParser(), {
          "context": context,                                                                                                                                                
          "question": question,
          "chat_history": chat_history,                                                                                                                                      
      }) 

def _answer_stream(context: str, question: str, chat_history: str = ""):
    """
    Same as _answer() but yields tokens one at a time instead of
    waiting for the full response.

    .invoke()  → waits 5 seconds → returns "The action items were..."
    .stream()  → yields "The" → " action" → " items" → " were" → ...

    This is a Python generator — it doesn't run until you iterate over it.
    Each iteration gives you one chunk of text from the LLM.
    """
    chain = ANSWER_PROMPT | llm | StrOutputParser()

    for chunk in chain.stream({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    }):
        yield chunk


### Router ----------------- 

def route_question(question: str, chat_history: str = "") -> str:                                                                                                          
      """Use LLM to decide which RAG strategy fits the question best."""
      strategy = _invoke_with_retry(ROUTER_PROMPT | llm | StrOutputParser(), {
          "question": question,                                                                                                                                              
          "chat_history": chat_history,
      })                                                                                                                                                                     
      strategy = strategy.strip().lower()

      valid = {"naive", "multi_query", "compression", "reject"}                                                                                                                
      return strategy if strategy in valid else "multi_query"

## Strategy 1: Naive RAG
# Direct embed → search → answer. No tricks, baseline.

def naive_rag(question: str, meeting_ids: list[str], chat_history: str = "") -> str:
      docs = retrieve(question, meeting_ids, k=5)
      return _answer(_format_docs(docs), question, chat_history)


# ── Strategy 3: Multi-Query ───────────────────────────────────────────────────
# Rephrase query 3 ways → search all 3 → union deduplicated results → answer.
# Catches relevant chunks that a single phrasing would miss.

def multi_query_rag(question: str, meeting_ids: list[str], chat_history: str = "") -> str:
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

      return _answer(_format_docs(all_docs[:8]), question, chat_history)

# ── Strategy 4: Contextual Compression ───────────────────────────────────────
# Retrieve wide (k=10) → LLM strips irrelevant sentences → answer with tight context.
# Best when transcript is long and noisy.

                                                                                                                                                                             
def contextual_compression_rag(question: str, meeting_ids: list[str], chat_history: str = "") -> str:
      docs = retrieve(question, meeting_ids, k=10)                                                                                                                           
      compressed = _invoke_with_retry(COMPRESSION_PROMPT | llm | StrOutputParser(), {                                                                                      
          "question": question,                                                                                                                                              
          "context": _format_docs(docs),
      })                                                                                                                                                                     
      return _answer(compressed, question, chat_history) 

