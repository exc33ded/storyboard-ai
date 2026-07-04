import os
from dotenv import load_dotenv

load_dotenv()

# --- Text / vision LLM (director, research summarization, narration, prompts) ---
# Uses the OpenAI SDK's chat.completions API. Point TEXT_BASE_URL at any
# OpenAI-compatible endpoint: Ollama (http://localhost:11434/v1), LM Studio
# (http://localhost:1234/v1), DeepSeek (https://api.deepseek.com), NVIDIA NIM,
# OpenRouter, Groq, or plain OpenAI (leave TEXT_BASE_URL unset).
TEXT_API_KEY = os.getenv("TEXT_API_KEY", "not-needed")
TEXT_BASE_URL = os.getenv("TEXT_BASE_URL") or None
TEXT_MODEL = os.getenv("TEXT_MODEL", "gpt-4o-mini")
# DeepSeek's "thinking mode" has no low/minimal setting (only high/max, or off) —
# disabling it entirely is the closest match to "minimal reasoning".
TEXT_EXTRA_BODY = {"thinking": {"type": "disabled"}} if TEXT_BASE_URL and "deepseek" in TEXT_BASE_URL else {}

# --- Vision LLM (narration polish, segmentation object-ID, reference-image
# verification) — DeepSeek has no vision support, so these 3 calls use a
# separate vision-capable model. Groq's free tier serves llama-4-scout
# (real image input) at no cost: https://console.groq.com/keys
VISION_API_KEY = os.getenv("VISION_API_KEY", "not-needed")
VISION_BASE_URL = os.getenv("VISION_BASE_URL", "https://api.groq.com/openai/v1")
VISION_MODEL = os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

# --- Image generation via Hugging Face Inference Providers ---
# ponytail: FLUX.2 [klein] is only routed as an image-EDITING model on HF's
# providers (no plain text-to-image mapping yet) — defaulting to FLUX.1-schnell
# (same lineage, Apache-2.0, has a real text-to-image route). Switch back to
# black-forest-labs/FLUX.2-klein-4B here once HF adds a t2i provider for it.
HF_API_KEY = os.getenv("HF_API_KEY")
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-schnell")

# Primary image gen: Puter (unofficial drivers API, browser SDK's own endpoint).
# Token: log in at puter.com → DevTools → Application → Local Storage → puter.auth.token
PUTER_AUTH_TOKEN = os.getenv("PUTER_AUTH_TOKEN")
# Comma-separated priority list — first model that works is used.
PUTER_IMAGE_MODELS = os.getenv(
    "PUTER_IMAGE_MODELS",
    "black-forest-labs/flux-schnell,gpt-image-1-mini,gemini-3.1-flash-image-preview",
)

# --- TTS: Kokoro (local, open-source, CPU-friendly) ---
# Measured speaking pace of Kokoro's default voices — used to size narration
# scripts against a target video length. Re-measure if you change voices.
TTS_WORDS_PER_MINUTE = int(os.getenv("TTS_WORDS_PER_MINUTE", "160"))
KOKORO_LANG_CODE = os.getenv("KOKORO_LANG_CODE", "a")  # "a" = American English
KOKORO_VOICE_SPEAKER_ONE = os.getenv("KOKORO_VOICE_SPEAKER_ONE", "af_heart")
KOKORO_VOICE_SPEAKER_TWO = os.getenv("KOKORO_VOICE_SPEAKER_TWO", "am_puck")

# --- Subtitles: faster-whisper (local, open-source) ---
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# --- Gemini (still used for optional Veo video gen) ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-2.5-pro"
VEO_MODEL = "veo-3.1-generate-preview"

# --- SearXNG (optional, self-hosted metasearch) ---
# If set, web research and reference-image search use your SearXNG instance
# instead of DuckDuckGo/Wikipedia — free, private, and not rate-limited.
# Quick setup: docker run -d -p 8888:8080 searxng/searxng
# then enable the JSON API in its settings.yml: search.formats: [html, json]
SEARXNG_URL = (os.getenv("SEARXNG_URL") or "").rstrip("/") or None

# SAM Segmentation Model URL
# To host SAM 3 on your own Cloud Run endpoint, follow instructions in sam3-hosting/README.md
# SAM_API_URL = "https://sam3-app-1040077537378.us-east4.run.app/predict"
SAM_API_URL = ""

if not TEXT_API_KEY or TEXT_API_KEY == "not-needed" and not TEXT_BASE_URL:
    print("WARNING: TEXT_API_KEY not set and no local TEXT_BASE_URL configured.")
if not PUTER_AUTH_TOKEN and not HF_API_KEY:
    print("WARNING: neither PUTER_AUTH_TOKEN nor HF_API_KEY set — image generation will fail.")
