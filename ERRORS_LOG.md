# RAGMeeting — Errors & Fixes Log

## 1. Cannot run backend with `python3 main.py`

**Problem:** Had to manually run `uvicorn main:app --reload` every time.

**Cause:** `main.py` was missing a `if __name__ == "__main__"` block.

**Fix:** Added to bottom of `backend/main.py`:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

---

## 2. Pydantic v1/v2 schema error — `_action_items` not declarable

**Error:**
```
ValueError: Value not declarable with JSON Schema, field: name='_action_items' type=ActionItem required=True
```

**Cause:** `llm.with_structured_output(MeetingSummary)` was called at module level in `rag/summarizer.py`. LangChain's Google GenAI integration internally converts Pydantic v2 models through Pydantic v1's schema generator. Pydantic v1 cannot handle `list[ActionItem]` (Python 3.10+ generic syntax) or nested Pydantic v2 models — it sees them as unknown types, prefixes the field name with `_`, and fails.

Changing `list[ActionItem]` to `List[ActionItem]` (from `typing`) and `str | None` to `Optional[str]` did NOT fix the issue because the v2→v1 model conversion itself was broken for nested models.

**Fix:** Upgraded all LangChain packages to latest versions which use Pydantic v2 natively:
```bash
pip install --upgrade langchain-google-genai langchain-core langchain langchain-chroma langchain-text-splitters langchain-community
```
Also moved `with_structured_output()` call from module-level into the `summarize_transcript()` function to avoid import-time failures.

---

## 3. Embedding model not found — 404 error

**Error:**
```
google.api_core.exceptions.NotFound: 404 models/gemini-embedding-004 is not found for API version v1beta
```

**Cause:** The embedding model name `models/gemini-embedding-004` in `.env` was wrong — that model doesn't exist.

**Fix:** Changed `EMBEDDING_MODEL` in `backend/.env` (line 6) to a valid model:
```
EMBEDDING_MODEL=models/gemini-embedding-001
```
Also updated the default in `backend/config/settings.py`.

**Note:** The `.env` file overrides `settings.py` defaults. We initially only fixed `settings.py` and the error persisted because `.env` still had the old value.

---

## 4. `ModuleNotFoundError: No module named 'langchain_core.beta'`

**Error:**
```
ModuleNotFoundError: No module named 'langchain_core.beta'
```

**Cause:** Stale `__pycache__` directories still referencing old `langchain_core` module paths after the package upgrade.

**Fix:** Cleared all Python cache and restarted:
```bash
cd backend && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; python3 main.py
```

---

## 5. 500 error on `/meetings/{id}/stop` — no traceback visible

**Problem:** Backend returned 500 but no error details were printed in the terminal.

**Cause:** The `except Exception as e` block in `api/meetings.py` caught the error and re-raised it as an `HTTPException`, but didn't log the traceback.

**Fix:** Added `traceback.print_exc()` inside the except block:
```python
except Exception as e:
    import traceback
    traceback.print_exc()
    meeting.status = "failed"
    db.commit()
    raise HTTPException(status_code=500, detail=str(e))
```

---

## 6. Streaming not visible on frontend — "Thinking..." forever

**Problem:** Backend logs showed tokens being sent, but the frontend only showed "Thinking..." and then the full answer appeared at once.

**Cause (two issues):**
1. `rag_stream()` in `query.py` was a sync `def` generator. FastAPI buffered the entire response before sending because the sync generator blocked the event loop.
2. Missing HTTP headers meant browsers/proxies could also buffer the SSE stream.

**Fix:**
- Changed `rag_stream()` to `async def` generator
- Used `asyncio.to_thread()` and `run_in_executor()` for sync LLM calls so the event loop stays free to flush each `yield`
- Added `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers to `StreamingResponse`

**Note:** Even after fixing, Gemini sends tokens in large chunks (1-2 per response) unlike OpenAI which streams word-by-word. So the streaming effect is less visible with Gemini.

---

## 7. Frontend SSE parsing breaking on partial chunks

**Problem:** `reader.read()` can deliver a chunk that splits an SSE event in half. `JSON.parse` would fail on incomplete data.

**Fix:** Added a `buffer` in `MeetingDetail.tsx` that accumulates data across chunks. Only complete events (terminated by `\n\n`) get parsed. Incomplete tails stay in the buffer for the next iteration.

```js
buffer += decoder.decode(value, { stream: true })
const parts = buffer.split('\n\n')
buffer = parts.pop()!  // keep incomplete tail
```

---

## 8. Non-questions hitting RAG pipeline — "Okay Got it" causing timeouts

**Problem:** Typing "Okay Got it", "thanks", "hello" etc. would trigger the full RAG pipeline, get routed to HyDE (vague query), make 2+ LLM calls, and timeout with "AI service is temporarily busy."

**Fix:** Added `reject` as a strategy option in the router prompt (`chain.py`). The router LLM now classifies greetings/chitchat/acknowledgments as `reject`. When `reject` is returned, `query.py` skips the entire pipeline and immediately responds with "Please ask a question about the meeting." — zero extra LLM calls since the router call was already happening.

---

## 9. Gemini 500 InternalServerError — 53 errors in one day

**Problem:** Google AI Studio dashboard showed 53x `500 InternalServerError` on April 9, 2026. No 429 (rate limit) errors — the issue was server-side failures.

**Cause:** `gemini-2.5-flash` is a **preview model** — not production-ready. Preview models have no uptime SLA and can fail randomly during Google's internal updates.

**Fix:** Switched to `gemini-2.0-flash` (GA/Generally Available model) in `config/settings.py`. GA models have stable uptime guarantees and fewer random failures.

---

## 10. HyDE strategy causing persistent failures

**Problem:** HyDE makes an extra LLM call to generate a hypothetical transcript excerpt before retrieval. If Gemini returns 500 errors, the retries (1s + 2s + 4s = 7s) fail and kill the entire stream.

**Fix:** Removed HyDE strategy entirely. Vague/short questions now route to `multi_query` instead, which rephrases the question 3 ways for better retrieval coverage without the fragile hypothesis generation step. Removed from: `chain.py` (prompt, function, valid set), `nodes.py` (import, strategy map, retrieve function).

---

## 11. Two database files in backend directory

**Problem:** Both `meetings.db` (0 bytes, empty) and `ragmeeting.db` (active) existed in the backend directory.

**Cause:** `meetings.db` was created by an earlier version of the code. The app uses `ragmeeting.db` (defined in `db/database.py` line 6).

**Fix:** `meetings.db` can be safely deleted — it's stale and unused.

---

# New Features Added

## A. Pipeline status events in streaming

Instead of showing a generic "Thinking..." spinner, the frontend now shows what the pipeline is actually doing:
- "Routing question..." → "Retrieving context..." → "Generating answer..." → answer appears

**Backend:** Added `status` SSE events before each pipeline step in `query.py`
**Frontend:** Added `status` field to `ChatMessage`, renders spinner with step name in chat bubble, removed hardcoded "Thinking..." spinner.

## B. Cache Augmented Generation (CAG)

Added a two-tier caching system to avoid repeated LLM calls for similar questions.

**Tier 1 — Exact match (free):** Normalizes the question (lowercase, strip punctuation, collapse spaces) and checks for an identical cached entry. Zero API calls.

**Tier 2 — Semantic similarity (1 embedding call):** If no exact match, embeds the question and compares against cached question embeddings using cosine similarity (threshold 0.85). Costs 1 API call but saves 2-3 LLM calls on a hit.

**Files:**
- `db/models.py` — Added `QueryCache` table with columns: id, meeting_id, question_raw, question_normalized, question_embedding, answer, strategy, created_at
- `rag/cache.py` — `normalize_question()`, `embed_question()`, `cosine_similarity()`, `get_cached_answer()`, `store_answer()`, `clear_meeting_cache()`
- `api/query.py` — Both `/query` and `/query/stream` check cache before running RAG pipeline. Cache hits in streaming are sent as a single token event.

## C. LLM provider switching (Gemini / Ollama)

Added ability to switch between Gemini and Ollama from the frontend without restarting the backend.

**Backend:**
- `config/settings.py` — Added `llm_provider` and `ollama_model` settings
- `rag/chain.py` — `_build_llm()` factory, `get_provider()`, `set_provider()` for runtime switching
- `api/settings_api.py` — New `GET/PUT /settings/provider` endpoints
- `main.py` — Registered settings router

**Frontend:**
- `MeetingDetail.tsx` — Toggle button (blue dot = Gemini, green dot = Ollama) next to chat header

**Dependencies:** Added `langchain-ollama>=0.3.0` to `requirements.txt`
