import requests 
from config.settings import settings 

HF_KOKORO_URL = "https://api-inference.huggingface.co/models/hexgrad/Kokoro-82M"


def text_to_speech(text: str) -> bytes:
    """
    Send answer text to HF Kokoro API and return raw audio bytes.
    Returns audio in flac format — frontend plays it directly.
    """
    response = requests.post(
        HF_KOKORO_URL,
        headers={"Authorization": f"Bearer {settings.hf_token}"},
        json={"inputs": text}
    )

    if response.status_code == 503:
        raise RuntimeError("Kokoro model is loading, please retry in 20 seconds")

    response.raise_for_status()
    return response.content