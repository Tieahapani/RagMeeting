import time
import logging
import requests
from config.settings import settings

logger = logging.getLogger(__name__)

HF_WHISPER_URL = "https://router.huggingface.co/hf-inference/models/openai/whisper-large-v3-turbo"


def _transcribe_chunk(audio_bytes: bytes, max_retries: int = 3) -> str:
    """Send a single audio chunk to HF Whisper API and return transcript text."""
    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "audio/webm",
    }

    logger.info(f"Whisper request: audio_bytes size = {len(audio_bytes)}, first 20 bytes = {audio_bytes[:20]}")

    for attempt in range(max_retries):
        response = requests.post(
            HF_WHISPER_URL,
            headers=headers,
            data=audio_bytes,
        )

        logger.info(f"Whisper response: status={response.status_code}, body={response.text[:500]}")

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


def transcribe_audio(audio_bytes: bytes, max_retries: int = 3) -> str:
    """Transcribe audio by sending directly to Whisper API (supports up to 25MB)."""
    return _transcribe_chunk(audio_bytes, max_retries)
