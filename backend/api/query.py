import uuid # to generate a unique id
import json as json_lib
import asyncio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta

from db.database import get_db 
from rag.cache import get_cached_answer, store_answer 
from db.models import Meeting 
from rag.graph import rag_graph 
from services.tts_service import text_to_speech

router = APIRouter(prefix="/query", tags=["query"])
audio_cache: dict[str, tuple[bytes, datetime]] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    meeting_id: str
    tts: bool = True  # optional — False if user just wants text, no audio


class QueryResponse(BaseModel):
    question: str
    answer: str
    strategy: str        # which RAG strategy the router picked
    audio_url: str | None # 
    cached: bool = False 


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/", response_model=QueryResponse)
async def query_meeting(
    request: QueryRequest, 
    db: Session = Depends(get_db)
): 
    # ---- Verify meeting exists and is ready ------ 
    meeting = db.query(Meeting).filter(Meeting.id == request.meeting_id).first()

    if not meeting: 
        raise HTTPException(status_code=404, detail="Meeting not found")
    
    if meeting.status != "processed":
        raise HTTPException(
            status_code=400, 
            detail=f"Meeting is not ready to query. Current status: {meeting.status}"
        )
    
    # ---- Check cache first --------
    cached = get_cached_answer(request.meeting_id, request.question, db)

    if cached:
        # Cache HIT — skip the entire RAG pipeline
        answer = cached["answer"]
        strategy = cached["strategy"]
    else:
        # Cache MISS — run full RAG pipeline
        try:
            result = rag_graph.invoke({
                "question": request.question,
                "meeting_ids": [request.meeting_id],
                "strategy": "",
                "answer": "",
                "chat_history": [],
            },
            config={"configurable": {"thread_id": f"meeting_{request.meeting_id}"}},
            )
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="AI service is temporarily unavailable. Please try again in a few seconds."
            )

        answer = result["answer"]
        strategy = result["strategy"]

        # Store in cache so next similar question is instant
        store_answer(request.meeting_id, request.question, answer, strategy, db)

    ## ----- Generate Audio + Cache It ------
    audio_url = None

    if request.tts:
        try:
            audio_bytes = text_to_speech(answer)
            cache_key = str(uuid.uuid4())
            expiry = datetime.utcnow() + timedelta(minutes=15)
            audio_cache[cache_key] = (audio_bytes, expiry)
            audio_url = f"/query/audio/{cache_key}"
        except Exception:
            audio_url = None

    return QueryResponse(
        question=request.question,
        answer=answer,
        strategy=strategy,
        audio_url=audio_url,
        cached=cached is not None,
    )   

@router.post("/stream")
async def query_meeting_stream(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Streaming version of /query.
    Sends tokens one at a time as Server-Sent Events (SSE).
    Frontend reads these and appends each token to the UI as it arrives.
    """
    # ---- Verify meeting exists and is ready ------
    meeting = db.query(Meeting).filter(Meeting.id == request.meeting_id).first()

    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    if meeting.status != "processed":
        raise HTTPException(
            status_code=400,
            detail=f"Meeting is not ready to query. Current status: {meeting.status}"
        )

    # ---- Check cache first --------
    cached = get_cached_answer(request.meeting_id, request.question, db)

    if cached:
        # Cache HIT — send full answer as one event
        def cached_stream():
            yield f"data: {json_lib.dumps({'type': 'strategy', 'data': cached['strategy']})}\n\n"
            yield f"data: {json_lib.dumps({'type': 'token', 'data': cached['answer']})}\n\n"
            yield f"data: {json_lib.dumps({'type': 'done', 'cached': True})}\n\n"

        return StreamingResponse(
            cached_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- Cache MISS — run RAG with streaming --------
    from rag.nodes import _retrieve_for_strategy, _format_history
    from rag.chain import route_question, _answer_stream

    def _event(data: dict) -> str:
        return f"data: {json_lib.dumps(data)}\n\n"

    async def rag_stream():
        try:
            # Step 1: Route the question
            yield _event({"type": "status", "data": "Routing question..."})
            history_str = _format_history([])
            strategy = await asyncio.to_thread(
                route_question, request.question, history_str
            )
            # Guardrail — router says this isn't a meeting question
            if strategy == "reject":
                yield _event({"type": "token", "data": "Please ask a question about the meeting."})
                yield _event({"type": "done", "cached": False})
                return

            yield _event({"type": "strategy", "data": strategy})

            # Step 2: Retrieve context
            yield _event({"type": "status", "data": "Retrieving context..."})
            context = await asyncio.to_thread(
                _retrieve_for_strategy,
                strategy, request.question, [request.meeting_id]
            )

            # Step 3: Stream the answer token by token
            yield _event({"type": "status", "data": "Generating answer..."})
            full_answer = ""
            loop = asyncio.get_event_loop()
            token_iter = _answer_stream(context, request.question, history_str)

            while True:
                token = await loop.run_in_executor(
                    None, lambda: next(token_iter, None)
                )
                if token is None:
                    break
                full_answer += token
                yield _event({"type": "token", "data": token})

            # Step 4: Cache the complete answer
            store_answer(
                request.meeting_id, request.question,
                full_answer, strategy, db
            )
            yield _event({"type": "done", "cached": False})

        except Exception as e:
            yield _event({"type": "error", "data": str(e)})

    return StreamingResponse(
        rag_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/audio/{cache_key}")
def get_audio(cache_key: str):
    """
    Serves cached TTS audio. Entry expires 15 mins after creation.
    User can replay as many times as they want within that window."""

    if cache_key not in audio_cache:
        raise HTTPException(status_code=404, detail="Audio not found or expired")

    audio_bytes, expiry = audio_cache[cache_key]

    if datetime.utcnow() > expiry:
        del audio_cache[cache_key]   # clean up expired entry
        raise HTTPException(status_code=404, detail="Audio has expired")

    return Response(content=audio_bytes, media_type="audio/flac")

