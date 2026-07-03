import os
import wave
import base64
from google import genai
from openai import OpenAI
from config import GEMINI_API_KEY, TEXT_API_KEY, TEXT_BASE_URL, VISION_API_KEY, VISION_BASE_URL

# Gemini client — still used for audio transcription and Veo video gen
client = None
if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)

# OpenAI-compatible text client — works with OpenAI, Ollama, LM Studio,
# DeepSeek, NVIDIA NIM, OpenRouter, Groq, etc. via TEXT_BASE_URL.
text_client = OpenAI(api_key=TEXT_API_KEY, base_url=TEXT_BASE_URL)

# Separate OpenAI-compatible client for the 3 vision calls (narration polish,
# segmentation object-ID, reference-image verification) — kept independent of
# text_client since not every text model/provider supports image input.
vision_client = OpenAI(api_key=VISION_API_KEY, base_url=VISION_BASE_URL)

# Groq's free tier is rate-limited (requests/tokens per minute and per day) —
# track the real limits from its response headers so the UI can show remaining
# quota instead of just failing mid-run. Hugging Face's Inference Providers
# free tier is credit-based rather than a simple per-call limit, so we only
# track a call count for it (no exact "remaining" figure is available per call).
usage_stats = {
    "groq": {"limit_requests": None, "remaining_requests": None, "limit_tokens": None, "remaining_tokens": None, "reset_requests": None, "reset_tokens": None},
    "hf_calls": 0,
}


def vision_chat(**kwargs):
    """Wraps vision_client.chat.completions.create, recording Groq's rate-limit headers."""
    raw = vision_client.chat.completions.with_raw_response.create(**kwargs)
    h = raw.headers
    usage_stats["groq"].update({
        "limit_requests": h.get("x-ratelimit-limit-requests"),
        "remaining_requests": h.get("x-ratelimit-remaining-requests"),
        "limit_tokens": h.get("x-ratelimit-limit-tokens"),
        "remaining_tokens": h.get("x-ratelimit-remaining-tokens"),
        "reset_requests": h.get("x-ratelimit-reset-requests"),
        "reset_tokens": h.get("x-ratelimit-reset-tokens"),
    })
    return raw.parse()


def record_hf_call():
    usage_stats["hf_calls"] += 1


def searxng_search(query: str, categories: str = "general", max_results: int = 8) -> list:
    """
    Queries a self-hosted SearXNG instance (SEARXNG_URL) via its JSON API.
    Returns the raw result dicts (keys: title, url, content, img_src for images).
    Raises on any failure — callers fall back to DuckDuckGo/Wikipedia.
    """
    import json as _json
    import urllib.request
    import urllib.parse
    from config import SEARXNG_URL
    if not SEARXNG_URL:
        raise RuntimeError("SEARXNG_URL not configured")
    params = urllib.parse.urlencode({"q": query, "format": "json", "categories": categories})
    req = urllib.request.Request(
        f"{SEARXNG_URL}/search?{params}",
        headers={"User-Agent": "StoryboardAI/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        data = _json.loads(response.read().decode("utf-8"))
    return data.get("results", [])[:max_results]


def image_to_data_url(image_path: str) -> str:
    """Reads an image file and returns it as a base64 data: URL for OpenAI vision messages."""
    mime_type = "image/png"
    if image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"

# Global Output Management
GLOBAL_OUTPUT_DIR = None

def set_output_dir(path: str):
    """Sets the directory where all tool outputs will be saved."""
    global GLOBAL_OUTPUT_DIR
    GLOBAL_OUTPUT_DIR = path
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    print(f"Output directory set to: {path}")

def _save_to_run_folder(content: str, filename: str, mode: str = "w"):
    """Helper to save content to the current run folder if enabled."""
    if GLOBAL_OUTPUT_DIR:
        full_path = os.path.join(GLOBAL_OUTPUT_DIR, filename)
        try:
            with open(full_path, mode, encoding="utf-8") as f:
                f.write(content)
            return full_path
        except Exception as e:
            print(f"Error saving to {filename}: {e}")
    return None

def save_pcm_to_wav(filename: str, pcm: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2):
    """
    Saves raw PCM data to a .wav file.
    """
    try:
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(rate)
            wf.writeframes(pcm)
        return True
    except Exception as e:
        print(f"Error saving wave file: {e}")
        return False

def get_video_duration(video_path: str) -> float:
    """
    Returns the duration of a video file in seconds.
    """
    import cv2
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        return duration
    except Exception as e:
        print(f"Error getting video duration: {e}")
        return 0.0
