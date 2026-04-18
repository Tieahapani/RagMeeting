import uuid 
from datetime import datetime 
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session 
from pydantic import BaseModel 

from db.database import get_db 
from db.models import Meeting 
from services.stt_service import transcribe_audio 
from rag.retriever import ingest_transcript 
from rag.summarizer import summarize_transcript, MeetingSummary
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.preprocessor import clean_transcript
from config.settings import settings 

router = APIRouter(prefix="/meetings", tags=["meetings"])

llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    google_api_key=settings.gemini_api_key,
    temperature=0
)

TITLE_PROMPT = ChatPromptTemplate.from_template("""
Generate a short meeting title (4-6 words) based on this transcript.
Return only the title, nothing else.

Transcript:
{transcript}
""")

## Response Schema 

class MeetingStartResponse(BaseModel):
    meeting_id: str
    started_at: str


class MeetingListItem(BaseModel):
    id: str
    title: str
    date: str
    duration: int
    status: str


class MeetingDetailResponse(BaseModel):
    id: str
    title: str
    date: str
    duration: int
    status: str
    summary: str | None
    key_points: list[str]
    action_items: list[dict]



# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/start", response_model=MeetingStartResponse)
def start_meeting(db: Session = Depends(get_db)):
    """
    Called when user hits Record.
    Creates meeting row in DB, returns meeting_id to frontend.
    Frontend stores this id and sends it back when stopping.
    """
    meeting_id = str(uuid.uuid4())
    now = datetime.utcnow()

    meeting = Meeting(
        id=meeting_id,
        title="Untitled Meeting",
        date=now,
        duration=0,
        transcript=None,
        status="recording"
    )
    db.add(meeting)
    db.commit()

    return MeetingStartResponse(
        meeting_id=meeting_id,
        started_at=now.isoformat()
    )

class MeetingStopAck(BaseModel):
    meeting_id: str
    status: str


def _process_meeting(meeting_id: str, audio_bytes: bytes):
    """Background task: transcribe, clean, summarize, ingest."""
    import json
    from db.database import SessionLocal

    db = SessionLocal()
    try:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting:
            return

        # Step 1: Transcribe + clean
        transcript = transcribe_audio(audio_bytes)
        transcript = clean_transcript(transcript)

        # Step 2: Generate title
        title = (TITLE_PROMPT | llm | StrOutputParser()).invoke({
            "transcript": transcript[:2000]
        })

        # Step 3: Ingest into Chroma
        ingest_transcript(transcript, meeting_id)

        # Step 4: Summarize
        summary: MeetingSummary = summarize_transcript(transcript)

        # Step 5: Update DB
        meeting.title = title.strip()
        meeting.transcript = transcript
        meeting.summary = summary.summary
        meeting.key_points = json.dumps(summary.key_points)
        meeting.action_items = json.dumps([
            {"task": item.task, "owner": item.owner, "due_date": item.due_date}
            for item in summary.action_items
        ])
        meeting.audio_data = None  # clear audio after successful processing
        meeting.status = "processed"
        db.commit()
        print(f"[OK] Meeting {meeting_id} processed successfully")

    except Exception as e:
        import traceback
        traceback.print_exc()
        meeting.status = "failed"
        db.commit()
        print(f"[FAIL] Meeting {meeting_id}: {e}")
    finally:
        db.close()


@router.post("/{meeting_id}/stop", response_model=MeetingStopAck)
async def stop_meeting(
    meeting_id: str,
    audio: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: Session = Depends(get_db)
):
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    # Save audio + update status immediately
    audio_bytes = await audio.read()
    duration = int((datetime.utcnow() - meeting.date).total_seconds())

    meeting.audio_data = audio_bytes
    meeting.duration = duration
    meeting.status = "processing"
    db.commit()

    # Kick off processing in background
    background_tasks.add_task(_process_meeting, meeting_id, audio_bytes)

    return MeetingStopAck(meeting_id=meeting_id, status="processing")


@router.post("/{meeting_id}/retry", response_model=MeetingStopAck)
def retry_meeting(
    meeting_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Retry processing a failed meeting using saved audio."""
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not meeting.audio_data:
        raise HTTPException(status_code=400, detail="No saved audio to retry")

    meeting.status = "processing"
    db.commit()

    background_tasks.add_task(_process_meeting, meeting_id, meeting.audio_data)
    return MeetingStopAck(meeting_id=meeting_id, status="processing")


@router.get("/", response_model=list[MeetingListItem])
def list_meetings(db: Session = Depends(get_db)):
    """
    Returns all processed meetings for the meeting list UI.
    Only shows processed meetings — not recording/processing/failed.
    """
    meetings = (
        db.query(Meeting)
        .filter(Meeting.status.in_(["processed", "failed", "processing"]))
        .order_by(Meeting.date.desc())
        .all()
    )

    return [
        MeetingListItem(
            id=m.id, 
            title=m.title, 
            date=m.date.isoformat(), 
            duration=m.duration, 
            status=m.status
        )

        for m in meetings 
    ]

@router.get("/{meeting_id}", response_model=MeetingDetailResponse)
def get_meeting(meeting_id: str, db: Session = Depends(get_db)):
    """Returns full meeting detail including summary, key points, and action items."""
    import json
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    return MeetingDetailResponse(
        id=meeting.id,
        title=meeting.title,
        date=meeting.date.isoformat(),
        duration=meeting.duration,
        status=meeting.status,
        summary=meeting.summary,
        key_points=json.loads(meeting.key_points) if meeting.key_points else [],
        action_items=json.loads(meeting.action_items) if meeting.action_items else [],
    )


    