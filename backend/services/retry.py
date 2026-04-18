import time
import random


def retry_on_rate_limit(fn, *args, max_attempts=5, **kwargs):
    """
    Call fn(*args, **kwargs) with exponential backoff on 429 / RESOURCE_EXHAUSTED.

    - 1st retry waits ~2s, 2nd ~4s, 3rd ~8s, 4th ~16s, 5th ~32s
    - Jitter (±0-1s) prevents thundering herd when multiple calls retry together
    - Only retries on rate limit errors; all other errors raise immediately
    """
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "resource_exhausted" in error_str

            if not is_rate_limit:
                raise  # not a rate limit error, fail immediately

            if attempt == max_attempts - 1:
                raise  # last attempt, give up

            wait = (2 ** (attempt + 1)) + random.uniform(0, 1)
            print(f"[RETRY] Gemini 429 — attempt {attempt + 1}/{max_attempts}, waiting {wait:.1f}s")
            time.sleep(wait)
