from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
from pydantic import BaseModel, Field

from config.settings import settings


# Output Schema
class ActionItem(BaseModel):
    task: str = Field(description="The action item or task to be completed")
    owner: Optional[str] = Field(default=None, description="Person responsible, None if not mentioned")
    due_date: Optional[str] = Field(default=None, description="Due date or timeframe, None if not mentioned")


class MeetingSummary(BaseModel):
    summary: str = Field(description="3-4 sentence overview of the entire meeting")
    key_points: List[str] = Field(description="Most important points discussed")
    action_items: List[ActionItem] = Field(description="All tasks assigned during the meeting")


# Prompt
SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
You are an expert meeting analyst. Analyze the meeting transcript below and extract:

1. A concise summary (3-4 sentences) covering what the meeting was about
2. The most important key points discussed (be specific, not generic)
3. All action items — who needs to do what, and by when if mentioned

Only include information that is explicitly stated in the transcript.
Do not infer or add anything not discussed.

Respond with valid JSON matching this schema:
{schema}

Transcript:
{transcript}
""")


# Main Function
def summarize_transcript(transcript: str) -> MeetingSummary:
    """Takes a raw meeting transcript and returns structured summary using with_structured_output."""
    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    )
    # ✅ Remove `method="json_mode"`; Gemini handles structured output via schema
    structured_llm = llm.with_structured_blank(MeetingSummary)
    chain = SUMMARY_PROMPT | structured_llm
    return chain.invoke({"transcript": transcript})