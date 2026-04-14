from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    gemini_api_key: str
    chroma_db_path: str = "./chroma_db"
    hf_token: str #This is the hugging face token 
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.0-flash"
    llm_provider: str = "gemini"       # "gemini" or "ollama"
    ollama_model: str = "llama3.1:8b"  # which Ollama model to use
    chunk_size: int = 400   # words per chunk
    chunk_overlap: int = 50  # words of overlap between chunks

    class Config:
        env_file = ".env"
        extra = "ignore"  # allow extra env vars like LANGCHAIN_* without errors


settings = Settings()
