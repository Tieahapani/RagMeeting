import json 
import math 
import re 
import logging 

from sqlalchemy.orm import Session 
from db.models import QueryCache
from rag.retriever import _query_embeddings
 # ^ Reusing your existing embedding function from retriever.py                                                                                                                                                
  # It uses task_type="retrieval_query" which is exactly what we need                                                                                                                                           
  # since we're embedding questions (queries), not documents  

logger = logging.getLogger(__name__)

#Normalize Function 

def normalize_question(question: str) -> str:                                                                                                                                                                 
      """         
      Strips a question down to its core for exact matching.
                                                                                                                                                                                                                
      "What were the ACTION ITEMS???" → "what were the action items"
      "  Give me a summary.  " → "give me a summary"                                                                                                                                                            
                                                                                                                                                                                                                
      This is tier 1 of our cache lookup — free, instant, no API calls.                                                                                                                                         
      Catches identical questions with different casing/punctuation.                                                                                                                                            
      """                                                                                                                                                                                                       
      text = question.lower().strip()
      # ^ lowercase everything and remove leading/trailing whitespace                                                                                                                                           
                                                                                                                                                                                                                
      text = re.sub(r'[^\w\s]', '', text)
      # ^ Remove ALL punctuation — periods, commas, question marks, etc.                                                                                                                                        
      # \w = word characters (letters, digits, underscore)                                                                                                                                                      
      # \s = whitespace
      # [^\w\s] = anything that's NOT a word char or whitespace = punctuation                                                                                                                                   
      # "what were the action items???" → "what were the action items"                                                                                                                                          
                                                                                                                                                                                                                
      text = re.sub(r'\s+', ' ', text)                                                                                                                                                                          
      # ^ Collapse multiple spaces into one                                                                                                                                                                     
      # "what  were   the" → "what were the"                                                                                                                                                                    
   
      return text    

# Embed Function 

def embed_question(question: str) -> list[float]:
      """                                                                                                                                                                                                       
      Converts a question string into a vector (list of floats).
                                                                                                                                                                                                                
      Uses the same Gemini embedding model your retriever uses,                                                                                                                                                 
      with task_type="retrieval_query" — optimized for questions.
                                                                                                                                                                                                                
      This costs 1 API call to Google, but it's much cheaper than                                                                                                                                               
      a full LLM call. Gemini embedding API is essentially free tier.                                                                                                                                           
      """                                                                                                                                                                                                       
      embeddings = _query_embeddings()
      # ^ Creates a GoogleGenerativeAIEmbeddings instance                                                                                                                                                       
      # Reuses your existing function so the model/config stays in one place
                                                                                                                                                                                                                
      vector = embeddings.embed_query(question)
      # ^ Sends the question to Gemini embedding API                                                                                                                                                            
      # Returns a list of floats like [0.012, -0.034, 0.078, ...]
      # For gemini-embedding-001 this is a 768-dimensional vector                                                                                                                                               
                                                                                                                                                                                                                
      return vector  

# Cosine Similarity 

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
      """                                                                                                                                                                                                       
      Measures how similar two vectors are. Returns a float between -1 and 1.
                                                                                                                                                                                                                
      1.0  = identical meaning                                                                                                                                                                                  
      0.75 = very similar (our threshold)
      0.0  = unrelated                                                                                                                                                                                          
                                                                                                                                                                                                                
      Formula: cos(θ) = (A · B) / (|A| × |B|)
                                                                                                                                                                                                                
      We compute this ourselves instead of importing numpy/scipy                                                                                                                                                
      because it's simple math and avoids adding a heavy dependency.
      """                                                                                                                                                                                                       
      dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
      # ^ Dot product: multiply each pair of elements, sum them all                                                                                                                                             
      # [1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 32
                                                                                                                                                                                                                
      magnitude_a = math.sqrt(sum(a * a for a in vec_a))
      # ^ Length of vector A: square each element, sum, square root                                                                                                                                             
      # |[1, 2, 3]| = √(1 + 4 + 9) = √14                                                                                                                                                                        
   
      magnitude_b = math.sqrt(sum(b * b for b in vec_b))                                                                                                                                                        
      # ^ Same for vector B
                                                                                                                                                                                                                
      if magnitude_a == 0 or magnitude_b == 0:
          return 0.0
      # ^ Edge case: if either vector is all zeros, can't divide by zero
      # This shouldn't happen with real embeddings but just in case                                                                                                                                             
                                                                                                                                                                                                                
      return dot_product / (magnitude_a * magnitude_b)                                                                                                                                                          
      # ^ The cosine similarity formula                                                                                                                                                                         
      # Normalizes the dot product by the magnitudes so the result                                                                                                                                              
      # is between -1 and 1 regardless of vector dimensions  

#Cache lookup (the main function)                                                                                                                                                                              
                  
def get_cached_answer(
      meeting_id: str,                                                                                                                                                                                          
      question: str,
      db: Session,                                                                                                                                                                                              
      threshold: float = 0.85
  ) -> dict | None:
      """
      Two-tier cache lookup:
                                                                                                                                                                                                                
      Tier 1: Exact normalized match (free, instant)
          "What were the action items?" matches "what were the action items"                                                                                                                                    
                                                                                                                                                                                                                
      Tier 2: Semantic similarity (1 embedding API call)
          "What were the action items?" matches "what tasks were assigned?"                                                                                                                                     
                  
      Returns dict with answer + strategy if hit, None if miss.                                                                                                                                                 
      threshold=0.85 means questions must be 85% semantically similar.
      """                                                                                                                                                                                                       
                  
      # ── Tier 1: Exact match ──────────────────────────────────────────────                                                                                                                                   
      normalized = normalize_question(question)
      # ^ "What were the ACTION ITEMS???" → "what were the action items"                                                                                                                                        
                                                                                                                                                                                                                
      exact_hit = (                                                                                                                                                                                             
          db.query(QueryCache)                                                                                                                                                                                  
          .filter(
              QueryCache.meeting_id == meeting_id,
              QueryCache.question_normalized == normalized,
          )
          .first()                                                                                                                                                                                              
      )
      # ^ SQL: SELECT * FROM query_cache                                                                                                                                                                        
      #        WHERE meeting_id = ? AND question_normalized = ?
      #        LIMIT 1
      # This is instant — no API calls, just a DB lookup                                                                                                                                                        
  
      if exact_hit:                                                                                                                                                                                             
          logger.info(f"Cache HIT (exact) for: {question[:50]}")
          return {                                                                                                                                                                                              
              "answer": exact_hit.answer,
              "strategy": exact_hit.strategy,                                                                                                                                                                   
              "cached": True,
          }                                                                                                                                                                                                     
      # ^ Found an exact match — return immediately, skip everything below
                                                                                                                                                                                                                
      # ── Tier 2: Semantic similarity ──────────────────────────────────────
      cached_entries = (                                                                                                                                                                                        
          db.query(QueryCache)                                                                                                                                                                                  
          .filter(QueryCache.meeting_id == meeting_id)
          .all()                                                                                                                                                                                                
      )           
      # ^ Get ALL cached Q&A pairs for this meeting
      # For a typical meeting this is maybe 10-50 entries — tiny
                                                                                                                                                                                                                
      if not cached_entries:
          return None                                                                                                                                                                                           
      # ^ No cache entries at all for this meeting — definite miss
                                                                                                                                                                                                                
      question_vector = embed_question(question)
      # ^ This is the ONE embedding API call we make                                                                                                                                                            
      # Convert the user's question into a vector for comparison                                                                                                                                                
  
      best_score = 0.0                                                                                                                                                                                          
      best_entry = None
                                                                                                                                                                                                                
      for entry in cached_entries:
          cached_vector = json.loads(entry.question_embedding)
          # ^ Deserialize the stored JSON string back to a list of floats
          # "[0.012, -0.034, ...]" → [0.012, -0.034, ...]                                                                                                                                                       
  
          score = cosine_similarity(question_vector, cached_vector)                                                                                                                                             
          # ^ How similar is this cached question to the new question?
                                                                                                                                                                                                                
          if score > best_score:                                                                                                                                                                                
              best_score = score
              best_entry = entry                                                                                                                                                                                
          # ^ Keep track of the most similar cached question
                                                                                                                                                                                                                
      if best_score >= threshold and best_entry:
          logger.info(                                                                                                                                                                                          
              f"Cache HIT (semantic, score={best_score:.3f}) for: {question[:50]}"
          )                                                                                                                                                                                                     
          return {
              "answer": best_entry.answer,                                                                                                                                                                      
              "strategy": best_entry.strategy,
              "cached": True,
          }                                                                                                                                                                                                     
      # ^ best_score >= 0.85 means the questions are similar enough
      # Return the cached answer                                                                                                                                                                                
                  
      logger.info(f"Cache MISS (best score={best_score:.3f}) for: {question[:50]}")                                                                                                                             
      return None 
      # ^ No cached question was similar enough — caller runs full RAG 

# Store in Cache 

def store_answer(
    meeting_id: str, 
    question: str, 
    answer: str, 
    strategy: str, 
    db: Session, 
) -> None: 
    """
    Stores a Q&A pair in the cache after a successful RAG pipeline run.
    Called only on cache MISS - after the full pipeline produces an answer,
    we store it so the next similar question gets a cache HIT.
    """
    normalized = normalize_question(question)

    embedding = embed_question(question)

    entry = QueryCache(
        meeting_id=meeting_id,
        question_raw=question,
        question_normalized=normalized,
        question_embedding=json.dumps(embedding),
        answer=answer,
        strategy=strategy,
    )

    db.add(entry)
    db.commit()

#Clear cache (for invalidation)

def clear_meeting_cache(meeting_id: str, db: Session) -> int:
      """
      Deletes all cached Q&A for a meeting.
   
      Called when a meeting is re-processed or deleted.
      Returns the number of entries deleted.
      """
      count = (
          db.query(QueryCache)
          .filter(QueryCache.meeting_id == meeting_id)
          .delete()
      )
      db.commit()
      return count
 