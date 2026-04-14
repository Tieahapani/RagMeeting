from datetime import datetime
from sqlalchemy import Integer, String, Text, DateTime
from sqlalchemy.orm import Mapped, mapped_column

from db.database import Base

class Meeting(Base):
    __tablename__ = "meetings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    title: Mapped[str] = mapped_column(String, nullable=False)
    date: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    duration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_points: Mapped[str | None] = mapped_column(Text, nullable=True)       # JSON string
    action_items: Mapped[str | None] = mapped_column(Text, nullable=True)     # JSON string
    status: Mapped[str] = mapped_column(String, nullable=False, default="recording")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Meeting id={self.id} title={self.title} status={self.status}>"


class QueryCache(Base):
    __tablename__ = "query_cache"
     # ^ Creates a table called "query_cache" in ragmeeting.db

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # ^ Auto-incrementing ID — we don't need UUIDs here, just a simple counter

    meeting_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
     # ^ Which meeting this Q&A belongs to                                                                                                                                                                     
      # index=True — makes lookups by meeting_id fast (SQLite creates a B-tree index)                                                                                                                           
      # We filter by meeting_id every time, so without an index it would scan every row  

    question_raw: Mapped[str] = mapped_column(Text, nullable=False)                                                                                                                                           
      # ^ The original question exactly as the user typed it                                                                                                                                                    
      # Stored for debugging — "what did the user actually ask?"  
    
    question_normalized: Mapped[str] = mapped_column(String, nullable=False)                                                                                                                                  
      # ^ Lowercased, stripped version — used for exact match (tier 1 lookup)                                                                                                                                   
      # "What were the ACTION ITEMS?" → "what were the action items" 

                                                                                                                                                                                                               
    question_embedding: Mapped[str] = mapped_column(Text, nullable=False)                                                                                                                                     
      # ^ The embedding vector stored as a JSON string — e.g. "[0.012, -0.034, ...]"                                                                                                                            
      # SQLite doesn't have an array type, so we serialize to JSON                                                                                                                                              
      # Used for semantic similarity (tier 2 lookup)   

    answer: Mapped[str] = mapped_column(Text, nullable=False)                                                                                                                                                 
      # ^ The LLM's answer — this is what we return on cache hit 

    strategy: Mapped[str] = mapped_column(String, nullable=False)                                                                                                                                             
      # ^ Which RAG strategy was used (naive, hyde, multi_query, compression)
      # Returned to frontend so it knows how the answer was generated     

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow
    )                                                                                                                                                                                                         
      # ^ When this cache entry was created — useful for TTL if we add it later                                                                                                                                                     
                                                       
    

