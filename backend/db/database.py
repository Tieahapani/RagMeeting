from sqlalchemy import create_engine #This creates a connection to your database
from sqlalchemy.orm import sessionmaker, DeclarativeBase #Sessionmaker a factory for creating sessions. A session is like a conversaton with the DB - you open it , do your read/write , then close it. 
# The base class all your models inherit from 


DATABASE_URL = "sqlite:///./ragmeeting.db"

engine = create_engine(
    DATABASE_URL, 
    connect_args={
        "check_same_thread": False, #When two people are trying to open the connection 
        "timeout": 30 

    }
)

#SessionLocal - It is a factory that creates DB sessions 
# Each request gets its own session, closed when request ends 
SessionLocal = sessionmaker(
    autocommit=False, # We control when changes are committed 
    autoflush=False,  # don't auto-write to DB until we say so 
    bind=engine 
)

# Base - all models inherit from this 
class Base(DeclarativeBase): 
    pass 

def init_db() -> None: 
    """Create all tables if they don't exist. Called once on app startup."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """
    FastAPI dependency - provides a DB session per request.
    Automatically closes session when request finishes. 
    """  
    db = SessionLocal()
    try: 
        yield db 
    finally: 
        db.close() 
        
             