from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config.settings import settings
from services.retry import retry_on_rate_limit

CLEAN_TRANSCRIPT_PROMPT = ChatPromptTemplate.from_template("""
You are a transcript editor. Your job is to clean up a raw speech-to-text transcript. 

  Rules:
  1. Add proper punctuation (periods, commas, question marks)
  2. Add speaker labels (Speaker 1, Speaker 2, etc.) when you detect a speaker change
  3. If a speaker's name is mentioned in the conversation (e.g., "I'll pass this to Rhea", "Right, David?"), replace their Speaker label with their actual name for ALL of their lines
  4. Add paragraph breaks between different topics or speakers
  5. Remove filler words (um, uh, hmm, like, you know) ONLY when they add no meaning
  6. Do NOT change, rephrase, or remove any meaningful words
  7. Do NOT summarize — keep the full transcript intact
  8. If you cannot detect speaker changes clearly, just add punctuation and paragraphs                                                                                                                          
                                                                                                                                                                                                                
  Raw transcript:                                                                                                                                                                                               
  {transcript}                                                                                                                                                                                                  
                                                                                                                                                                                                                
  Cleaned transcript:
  """)    


def clean_transcript(transcript: str) -> str: 
    """
    Takes raw  whisper output and returns a cleaned version with punctuation, speaker labels, and pargraph breaks. 
    This runs ONCE per meeting, right after transcription. 
    The cleaned version is what gets: 
      - stored in the DB (so the user sees clean text)
      - chunked and embedded into Chroma (so retrieval works better)
      - fed to the summarizer (so summarize are more acccurate)
    """

    llm = ChatGoogleGenerativeAI(
        model=settings.llm_model, 
        google_api_key=settings.gemini_api_key, 
        temperature=0, 

    )

    chain = CLEAN_TRANSCRIPT_PROMPT | llm

    result = retry_on_rate_limit(chain.invoke, {"transcript": transcript})
    return result.content 