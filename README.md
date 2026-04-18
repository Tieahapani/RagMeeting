##RAGMeeting
An AI-powered meeting assistant that records, transcribes, summarizes, and lets you ask questions about your meetings using Retrieval-Augmented Generation (RAG).

##What Is This Project?
RAGMeeting is a full-stack application where you can record a meeting directly in the browser, transcribe the audio using OpenAI Whisper, clean the transcript with speaker labels and punctuation via Gemini, generate a structured summary with key points and action items, and ask natural language questions about the meeting — all powered by a multi-strategy RAG pipeline with streaming responses.

##Architecture
Backend (Python / FastAPI)
The backend runs on Render using FastAPI + Gunicorn and handles two main pipelines:
Meeting Processing Pipeline — triggered when a user stops recording. Processing runs entirely in the background so the API returns immediately:

Audio Upload → Transcribe (Whisper) → Clean Transcript (Gemini) → Generate Title (Gemini) → Ingest into PGVector → Summarize (Gemini) → Mark as Processed

Each step saves its result to Neon PostgreSQL immediately. If a later step fails, earlier results are preserved. Users can retry failed meetings without re-recording.
RAG Query Pipeline — built with LangGraph and uses an adaptive multi-strategy approach:

Router — Gemini classifies each question into a strategy: naive (direct search), multi_query (rephrases question 3 ways and deduplicates), compression (retrieves wide then extracts relevant sentences), or reject (filters chitchat)
Hybrid Retrieval — combines PGVector semantic search + BM25 keyword search merged via Reciprocal Rank Fusion (50/50 weighting)
Streaming Response — tokens stream to the frontend via Server-Sent Events (SSE)
Two-Tier Query Cache — Tier 1 is exact normalized text match (free), Tier 2 is cosine similarity on embeddings (threshold 0.85)

##Key backend tools: FastAPI, PostgreSQL + PGVector (Neon), LangChain + LangGraph, Gemini 2.0 Flash, Gemini Embedding 001, Hugging Face Whisper Large v3 Turbo, BM25, LangSmith
##Frontend (React / TypeScript)
The frontend is hosted on Vercel and built with React 19 + TypeScript, Vite, Tailwind CSS v4, and React Router v7. It communicates with the backend via REST for CRUD operations and SSE for streaming chat responses. The UI polls the backend every 3 seconds during meeting processing and shows a progress indicator until the pipeline completes.

##Problems Faced & How I Solved Them
1. Low Retrieval Precision — Fixed with Hybrid Search
Problem: The initial retrieval used only semantic search (PGVector embeddings). For meeting transcripts, this missed relevant chunks when users phrased questions differently than the transcript wording. For example, asking "what tasks were assigned?" would miss chunks containing "action items" because embeddings don't capture exact keyword overlap.
Fix: Implemented hybrid retrieval combining semantic search (PGVector) and keyword search (BM25), merged using Reciprocal Rank Fusion so chunks appearing in both searches rank highest. Also switched to speaker-aware chunking — splitting transcripts on speaker labels instead of arbitrary character boundaries — so each chunk preserves a complete speaker turn with full context. Combined, these changes improved context precision by ~40% in the evaluation pipeline.

2. Gemini 429 Rate Limit Errors
Problem: The free tier Gemini API has strict rate limits (~15 requests/minute). During meeting processing, the pipeline makes multiple Gemini calls back-to-back (cleaning → title → summarization), which exhausted the limit and crashed the entire pipeline with no recovery.
Fix: Built a centralized retry_on_rate_limit() utility with exponential backoff + jitter (2s → 4s → 8s → 16s → 32s, with 0–1s random jitter to prevent thundering herd). Non-rate-limit errors fail immediately. Also added the two-tier query cache so repeated or semantically similar questions skip Gemini entirely, and added a graceful fallback title of "Meeting on {date}" if Gemini is fully unavailable.

3. Processing Timeout — 500 Error on Long Recordings
Problem: Any meeting longer than ~30 seconds would return a 500 Internal Server Error. Users saw a blank error screen and their recording was lost.
Root Cause: The entire processing pipeline ran synchronously inside the HTTP request handler. Gunicorn's default worker timeout is 30 seconds, so the worker was killed (SIGTERM) before processing finished.
Fix: Redesigned the pipeline to use FastAPI BackgroundTasks with progressive saving. The /stop endpoint now saves the raw audio to the database and returns immediately with status: "processing". All heavy work runs in the background, with each step saving its result independently. Raw audio is preserved in the database so users can hit "Retry Processing" on failed meetings without re-recording. Gunicorn timeout was also increased to 300 seconds as an additional safety net
