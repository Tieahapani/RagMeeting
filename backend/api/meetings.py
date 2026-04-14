import uuid 
from datetime import datetime 
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
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


class MeetingStopResponse(BaseModel):
    meeting_id: str
    title: str
    duration: int
    summary: str
    key_points: list[str]
    action_items: list[dict]


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

@router.post("/{meeting_id}/stop", response_model=MeetingStopResponse)
async def stop_meeting(
    meeting_id: str, 
    audio: UploadFile = File(...), 
    db: Session = Depends(get_db)
): 
    # ── Fetch meeting from DB ─────────────────────────────────────────────────
    meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
    # query Meeting table, find row where id matches, return first result
    
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
        # if no meeting found with this id, return 404 to frontend immediately

    # ── Calculate duration ────────────────────────────────────────────────────
    duration = int((datetime.utcnow() - meeting.date).total_seconds())
    # current time minus when recording started = how long the meeting was
    # .total_seconds() gives float like 2700.0 → int() converts to 2700

    # ── Update status to processing ───────────────────────────────────────────
    meeting.status = "processing"           # tells UI heavy work is happening
    meeting.duration = duration             # save duration even before processing
    db.commit()                             # write these two changes to disk now
    
    try: 
        # ── Step 1: Transcribe ────────────────────────────────────────────────
        audio_bytes = await audio.read()    # read entire audio file into memory as bytes
        transcript = transcribe_audio(audio_bytes)
        transcript = clean_transcript(transcript)
        # send bytes to HF Whisper API → returns transcript string

        # ── Step 2: Generate title ────────────────────────────────────────────
        title = (TITLE_PROMPT | llm | StrOutputParser()).invoke({
            "transcript": transcript[:2000] # only first 2000 chars needed for title
        })
        # LCEL chain: prompt → LLM → parse as string → returns "Q3 Budget Planning"
        

        # ── Step 3: Ingest into Chroma ────────────────────────────────────────
        ingest_transcript(transcript, meeting_id)
        # splits transcript into chunks → embeds → stores in Chroma
        # uses meeting_id as collection name so chunks are scoped to this meeting 

        
        # ── Step 4: Summarize ─────────────────────────────────────────────────
        summary: MeetingSummary = summarize_transcript(transcript)
        # sends full transcript to Gemini → returns MeetingSummary Pydantic object
        # MeetingSummary has: summary, key_points, action_items

        # ── Step 5: Update DB record ──────────────────────────────────────────
        import json
        meeting.title = title.strip()
        meeting.transcript = transcript
        meeting.summary = summary.summary
        meeting.key_points = json.dumps(summary.key_points)
        meeting.action_items = json.dumps([
            {"task": item.task, "owner": item.owner, "due_date": item.due_date}
            for item in summary.action_items
        ])
        meeting.status = "processed"
        db.commit()

        # Return response to frontend 
        return MeetingStopResponse(
            meeting_id=meeting_id, 
            title=meeting.title, 
            duration=duration, 
            summary=summary.summary,        # The team discussed Q3 planning...."
            key_points=summary.key_points, 
            action_items=[
                {
                    "task": item.task, 
                    "owner": item.owner, 
                    "due_date": item.due_date
                }

                for item in summary.action_items  
            ]
        ) 
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        meeting.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/", response_model=list[MeetingListItem])
def list_meetings(db: Session = Depends(get_db)):
    """
    Returns all processed meetings for the meeting list UI.
    Only shows processed meetings — not recording/processing/failed.
    """
    meetings = (
        db.query(Meeting)
        .filter(Meeting.status == "processed")
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


    