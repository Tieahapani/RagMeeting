from contextlib import asynccontextmanager
from dotenv import load_dotenv

load_dotenv()  # exports .env vars (like LANGCHAIN_*) into OS environment

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.database import init_db
from api.meetings import router as meetings_router
from api.query import router as query_router
from api.settings_api import router as settings_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="RAGMeeting API", lifespan=lifespan)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",       # local dev
        "https://rag-meeting-afwu.vercel.app",  # production frontend
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(meetings_router)  # /meetings/*
app.include_router(query_router)     # /query/*
app.include_router(settings_router)  # /settings/*

# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
