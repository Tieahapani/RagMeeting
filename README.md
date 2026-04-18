 RAGMeeting

  An AI-powered meeting assistant that records,
  transcribes, summarizes, and lets you ask
  questions about your meetings using
  Retrieval-Augmented Generation (RAG).

  What It Does

  RAGMeeting is a full-stack application where you
  can:

  1. Record a meeting directly in the browser
  2. Transcribe the audio using OpenAI Whisper (via
   Hugging Face)
  3. Clean the transcript — adds punctuation,
  speaker labels, and paragraph breaks using Gemini
  4. Summarize the meeting — generates a summary,
  key points, and action items
  5. Ask questions about the meeting — uses a
  multi-strategy RAG pipeline with streaming
  responses

  Tech Stack

  Backend (Python/FastAPI)

  - FastAPI — REST API with SSE streaming
  - PostgreSQL (Neon) — meeting storage
  - PGVector — vector embeddings for semantic
  search
  - LangChain + LangGraph — RAG pipeline
  orchestration
  - Gemini 2.0 Flash — LLM for routing, answering,
  summarization, and transcript cleaning
  - Gemini Embedding 001 — document and query
  embeddings
  - Hugging Face Whisper Large v3 Turbo —
  speech-to-text
  - BM25 — keyword search for hybrid retrieval
  - Ollama (local only) — optional local LLM
  alternative (Llama 3.1 8B)

  Frontend (React/TypeScript)

  - React 19 + TypeScript
  - Vite — build tool
  - Tailwind CSS v4 — styling
  - React Router v7 — navigation

  Infrastructure

  - Render — backend hosting
  - Vercel — frontend hosting
  - Neon — managed PostgreSQL
  - LangSmith — tracing and evaluation

  Architecture

  ┌────────────────────────────────────────────────
  ─────┐
  │  Frontend (Vercel)
       │
  │  React + TypeScript + Tailwind
       │
  │
       │
  │  ┌──────────┐  ┌──────────────┐
  ┌───────────────┐  │
  │  │ Record   │  │ Meeting List │  │ Meeting
  Detail│  │
  │  │ Meeting  │  │              │  │ + Chat (SSE)
    │  │
  │  └──────────┘  └──────────────┘
  └───────────────┘  │
  └─────────────────────┬──────────────────────────
  ─────┘
                        │ REST API + SSE
  ┌─────────────────────▼──────────────────────────
  ─────┐
  │  Backend (Render)
       │
  │  FastAPI + Gunicorn
       │
  │
       │
  │
  ┌─────────────────────────────────────────────┐
    │
  │  │ Meeting Processing Pipeline (Background)
   │    │
  │  │ Whisper → Clean → Title → Ingest → Summarize
   │    │
  │
  └─────────────────────────────────────────────┘
    │
  │
       │
  │
  ┌─────────────────────────────────────────────┐
    │
  │  │ RAG Query Pipeline (LangGraph)
   │    │
  │  │ Router → Retrieve (Hybrid) → Stream Answer
   │    │
  │  │
    │    │
  │  │ Strategies: naive | multi_query |
  compression │    │
  │
  └─────────────────────────────────────────────┘
    │
  │
       │
  │  ┌────────────┐  ┌─────────────┐
       │
  │  │ Query Cache│  │ Retry Logic │
       │
  │  │ (2-tier)   │  │ (exp backoff)│
        │
  │  └────────────┘  └─────────────┘
       │
  └─────────────────────┬──────────────────────────
  ─────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
     PostgreSQL     PGVector      Gemini API
     (Neon)         (Neon)        (Google AI)

  RAG Pipeline

  The query pipeline uses an adaptive
  multi-strategy RAG approach:

  1. Router — Gemini classifies each question into
  a strategy:
    - naive — direct embed-search-answer for
  specific factual questions
    - multi_query — rephrases the question 3 ways,
  searches all, deduplicates results (for
  broad/vague questions)
    - compression — retrieves wide (k=10), then LLM
   extracts only relevant sentences (for long noisy
   transcripts)
    - reject — filters out non-meeting questions
  (greetings, chitchat)
  2. Hybrid Retrieval — combines semantic search
  (PGVector) + keyword search (BM25) using
  Reciprocal Rank Fusion (50/50 weighting)
  3. Streaming Response — tokens stream to the
  frontend via Server-Sent Events (SSE) as Gemini
  generates them
  4. Two-Tier Query Cache — avoids redundant LLM
  calls:
    - Tier 1: Exact normalized text match (free,
  instant)
    - Tier 2: Cosine similarity on embeddings
  (threshold 0.85, costs 1 embedding call)

  Meeting Processing Pipeline

  When a user stops recording, processing happens
  in the background:

  Audio Upload → Save audio to DB → Return
  immediately (status: "processing")
                      │
                      ▼ (Background Task)
                Step 1: Transcribe (Whisper API) →
  save transcript
                Step 2: Clean transcript (Gemini) →
   save cleaned version
                Step 3: Generate title (Gemini) →
  save title (fallback: "Meeting on {date}")
                Step 4: Ingest into PGVector →
  chunk + embed
                Step 5: Summarize (Gemini) → save
  summary, key points, action items
                Step 6: Clear audio data → mark
  status "processed"

  Progressive saving — each step saves to the
  database immediately. If Gemini fails at step 5,
  you still have the transcript from step 1.

  Retry on failure — if processing fails, the raw
  audio is preserved in the database. Users can
  click "Retry Processing" to re-run the entire
  pipeline.

  Chunking Strategy

  Transcripts are split using a speaker-aware
  chunking strategy:

  1. First, try to split on speaker labels (Speaker
   1:, Dr. Lee:, etc.)
  2. If a single speaker turn exceeds the chunk
  size (400 words), sub-split using
  RecursiveCharacterTextSplitter
  3. If no speaker labels are detected, fall back
  to fixed-size character splitting with 50-word
  overlap

  Evaluation Pipeline

  The project includes a RAG evaluation framework
  using GPT-4o as a judge:

  - Golden dataset — 10 human-verified Q&A pairs
  from a real medical emergency meeting, stored in
  LangSmith
  - 4 metrics scored per question:
    - Faithfulness — is the answer grounded in the
  retrieved contexts?
    - Answer Relevancy — does the answer address
  the question?
    - Context Precision — are the retrieved chunks
  relevant?
    - Context Recall — do the chunks contain enough
   info for the ground truth answer?
  - Results pushed to LangSmith for tracking across
   experiments

  cd backend
  python eval/create_dataset.py   # upload golden
  dataset (one-time)
  python eval/run_eval.py          # run evaluation

  Project Structure

  RAGMeeting/
  ├── backend/
  │   ├── main.py                  # FastAPI app
  entry point
  │   ├── requirements.txt
  │   ├── config/
  │   │   └── settings.py          # Pydantic
  settings (env vars)
  │   ├── api/
  │   │   ├── meetings.py          # Meeting CRUD +
   background processing
  │   │   ├── query.py             # RAG query +
  streaming endpoint
  │   │   └── settings_api.py      # LLM provider
  toggle
  │   ├── db/
  │   │   ├── database.py          # SQLAlchemy
  engine + session
  │   │   └── models.py            # Meeting +
  QueryCache models
  │   ├── rag/
  │   │   ├── chain.py             # LLM prompts,
  strategies, streaming
  │   │   ├── graph.py             # LangGraph
  state machine
  │   │   ├── nodes.py             # Router + RAG
  graph nodes
  │   │   ├── state.py             # RAGState
  TypedDict
  │   │   ├── retriever.py         # Hybrid
  retrieval (PGVector + BM25 + RRF)
  │   │   ├── cache.py             # Two-tier query
   cache
  │   │   ├── preprocessor.py      # Transcript
  cleaning (Gemini)
  │   │   └── summarizer.py        # Structured
  summarization (Gemini)
  │   ├── services/
  │   │   ├── stt_service.py       # Whisper API
  (Hugging Face)
  │   │   └── retry.py             # Exponential
  backoff for rate limits
  │   └── eval/
  │       ├── create_dataset.py    # Upload golden
  dataset to LangSmith
  │       └── run_eval.py          # Run RAG
  evaluation with GPT-4o judge
  ├── frontend/
  │   ├── src/
  │   │   ├── App.tsx              # Router setup
  │   │   ├── config.ts            # API base URL
  │   │   └── pages/
  │   │       ├── RecordMeeting.tsx # Audio
  recording + upload
  │   │       ├── MeetingList.tsx   # Meeting list
  with status badges
  │   │       └── MeetingDetail.tsx # Summary +
  chat interface
  │   └── package.json
  └── ERRORS_LOG.md

  Setup

  Prerequisites

  - Python 3.10+
  - Node.js 18+
  - PostgreSQL with pgvector extension (or Neon)

  Environment Variables

  Copy backend/.env.example to backend/.env and
  fill in:

  GEMINI_API_KEY=your-gemini-api-key
  HF_TOKEN=your-huggingface-token
  OPENAI_API_KEY=your-openai-api-key          # for
   eval pipeline only
  DATABASE_URL=postgresql://...               #
  Neon connection string
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=your-langsmith-api-key
  LANGCHAIN_PROJECT=RagMeeting

  Backend

  cd backend
  pip install -r requirements.txt
  python main.py

  Frontend

  cd frontend
  npm install
  npm run dev

  Deployment

  - Backend: Deployed on Render (web service)
    - Build command: pip install -r
  requirements.txt
    - Start command: gunicorn main:app -w 2 -k
  uvicorn.workers.UvicornWorker --bind
  0.0.0.0:$PORT --timeout 300
  - Frontend: Deployed on Vercel (auto-deploys from
   GitHub)
  - Database: Neon PostgreSQL (free tier)

  Major Problems Faced & Solutions

  1. Low Retrieval Precision — Improved by 40% with
   Hybrid Search

  Problem: The initial retrieval used only semantic
   search (PGVector embeddings). For meeting
  transcripts, this often missed relevant chunks —
  especially when users asked questions using
  different words than what was in the transcript.
  For example, asking "what tasks were assigned?"
  would miss chunks containing "action items"
  because the embeddings didn't capture the exact
  keyword overlap.

  Root Cause: Pure semantic search struggles with
  domain-specific terminology and exact keyword
  matches in conversational transcripts. A user
  asking about "the BP reading" needs exact keyword
   matching, not just semantic similarity.

  Fix: Implemented hybrid retrieval combining two
  search methods merged with Reciprocal Rank Fusion
   (RRF):
  - Semantic search (PGVector) — finds conceptually
   similar chunks using Gemini embeddings
  - Keyword search (BM25) — finds exact lexical
  matches using term frequency
  - RRF fusion — merges both ranked lists with
  50/50 weighting, so chunks that appear in both
  searches rank highest

  This also included speaker-aware chunking —
  splitting transcripts on speaker labels (Dr.
  Lee:, Speaker 1:) instead of arbitrary character
  boundaries, so each chunk preserves a complete
  speaker turn with full context. Combined, these
  changes improved context precision by ~40% in our
   evaluation pipeline.

  2. Gemini 429 Rate Limit Errors

  Problem: The free tier Gemini API has strict rate
   limits (~15 requests per minute). During meeting
   processing, the pipeline makes multiple Gemini
  calls back-to-back (transcript cleaning → title
  generation → summarization), which would exhaust
  the rate limit and crash the entire pipeline.

  Root Cause: No retry logic — a single 429
  response would kill the pipeline and lose all
  progress.

  Fix: Built a centralized retry_on_rate_limit()
  utility with exponential backoff + jitter:
  - Retries on 429/RESOURCE_EXHAUSTED errors with
  increasing delays: 2s → 4s → 8s → 16s → 32s
  - Adds random jitter (0-1s) to prevent thundering
   herd when multiple calls retry simultaneously
  - Non-rate-limit errors fail immediately (no
  pointless retries)
  - Applied across all Gemini call sites:
  transcript cleaning, title generation,
  summarization
  - Added a two-tier query cache so
  repeated/similar questions hit the cache instead
  of Gemini
  - Title generation has a graceful fallback to
  "Meeting on {date}" if Gemini is completely
  unavailable

  3. Meeting Processing Timeout — 500 Error on Long
   Recordings

  Problem: After stopping a recording, the backend
  would return a 500 Internal Server Error for any
  meeting longer than ~30 seconds of audio. The
  user saw a blank error screen and their meeting
  was lost.

  Root Cause: The entire processing pipeline
  (Whisper transcription → Gemini cleaning →
  embedding ingestion → Gemini summarization) ran
  synchronously inside the HTTP request handler.
  Gunicorn's default worker timeout is 30 seconds,
  so the worker was killed (SIGTERM) before
  processing could finish, resulting in a 500
  error.

  Fix: Redesigned the pipeline to use background
  processing with progressive saving:
  - The stop endpoint now saves the raw audio to
  the database and returns immediately with status:
   "processing"
  - All heavy processing (transcribe → clean →
  title → ingest → summarize) runs in a FastAPI
  BackgroundTask
  - Each step saves its result to the database
  immediately — if step 5 fails, you still have the
   transcript from step 1
  - The raw audio is preserved in the database so
  users can retry failed processing without
  re-recording
  - Frontend polls the backend every 3 seconds and
  shows a progress indicator until processing
  completes
  - Failed meetings show a red banner with a "Retry
   Processing" button and the saved transcript
  - Increased Gunicorn timeout to 300 seconds as an
   additional safety net

  Live URLs

  - Frontend: https://rag-meeting-afwu.vercel.app
  - Backend: Hosted on Render (health check at
  /health)  
