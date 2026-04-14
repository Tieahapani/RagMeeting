import time
import io 
import logging
import requests 
from pydub import AudioSegment
from config.settings import settings

logger = logging.getLogger(__name__)

HF_WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"

SEGMENT_DURATION_MS = 10 * 60 * 1000 


def _transcribe_chunk(audio_bytes: bytes, max_retries: int = 3) -> str:
    """Send a single audio chunk to HF Whisper API and return transcript text."""
    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "audio/webm",
    }

    for attempt in range(max_retries):
        response = requests.post(
            HF_WHISPER_URL,
            headers=headers,
            data=audio_bytes,
        )

        if response.status_code == 503:
            wait = 20 if attempt == 0 else 10
            logger.warning(f"Whisper model loading, retry {attempt + 1}/{max_retries} in {wait}s")
            time.sleep(wait)
            continue

        if response.status_code >= 500 and attempt < max_retries - 1:
            wait = 2 ** attempt
            logger.warning(f"Whisper API error {response.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()
        data = response.json()
        transcript = data.get("text", "").strip()

        if not transcript:
            raise RuntimeError("Whisper returned empty transcript")

        return transcript

    raise RuntimeError(f"Whisper API failed after {max_retries} attempts")


def _split_audio(audio_bytes: bytes) -> list[bytes]: 
    """Split audio into segments of SEGMENT_DURATION_MS. Uses pydub to detect format, split, and export each segment as webm."""
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    duration_ms = len(audio)

    # Short audio - no splititng needed 
    if duration_ms <= SEGMENT_DURATION_MS: 
        return [audio_bytes]
    segments = []
    for start in range(0, duration_ms, SEGMENT_DURATION_MS): 
        end = min(start + SEGMENT_DURATION_MS, duration_ms)
        chunk = audio[start:end]

        buffer = io.BytesIO()
        chunk.export(buffer, format="webm")
        segments.append(buffer.getvalue())

    logger.info(f"Split {duration_ms/1000:.0f}s audio into {len(segments)} segments")  
    return segments

def transcribe_audio(audio_bytes: bytes, max_retries: int = 3) -> str:
    """Transcribe audio of any length, splitting long audio into segments."""
    segments = _split_audio(audio_bytes)

    if len(segments) == 1:
        return _transcribe_chunk(audio_bytes, max_retries)

    transcripts = []
    for i, segment in enumerate(segments):
        logger.info(f"Transcribing segment {i+1}/{len(segments)}")
        text = _transcribe_chunk(segment, max_retries)
        if text:
            transcripts.append(text)

    full_transcript = " ".join(transcripts)

    if not full_transcript.strip():
        raise RuntimeError("Whisper returned empty transcript for all segments")

    return full_transcript
