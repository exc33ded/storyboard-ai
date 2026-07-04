import os
import re
import time
import numpy as np
import soundfile as sf
from config import KOKORO_LANG_CODE, KOKORO_VOICE_SPEAKER_ONE, KOKORO_VOICE_SPEAKER_TWO
from . import utils

# Kokoro pipeline is loaded lazily since it loads model weights on first use.
_pipeline = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        _pipeline = KPipeline(lang_code=KOKORO_LANG_CODE)
    return _pipeline


def _synthesize(text: str, voice: str, speed: float = 1.0) -> np.ndarray:
    pipeline = _get_pipeline()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=speed)]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


SAMPLE_RATE = 24000
_CUE_RE = re.compile(r"\[([^\]]{1,40})\]")


def _cue_effect(name: str):
    """Map a bracketed stage cue to (silence_seconds, volume, speed_multiplier).
    volume/speed apply to the text chunk that follows the cue, until the next cue."""
    name = name.lower()
    if "long pause" in name:
        return 1.0, None, None
    if "pause" in name or "beat" in name or "silence" in name:
        return 0.6, None, None
    if any(w in name for w in ("softly", "quietly", "whisper", "gently", "hushed")):
        return 0.15, 0.55, 0.92
    if any(w in name for w in ("slowly", "solemn", "somber", "gravely")):
        return 0.0, None, 0.85
    if any(w in name for w in ("quickly", "fast", "urgent", "excited", "energetic")):
        return 0.0, None, 1.12
    # unknown cue: don't speak it, just take a small breath
    return 0.3, None, None


def _synthesize_with_cues(text: str, voice: str, speed: float) -> np.ndarray:
    """Synthesize text, honoring [pause]/[softly]/[slowly]-style stage cues:
    pauses become real silence; delivery cues adjust volume/speed of the
    following chunk (reset at the next cue). Cues are never spoken."""
    segments = []
    volume, spd = 1.0, speed
    parts = _CUE_RE.split(text)  # [text, cue, text, cue, text, ...]
    for i, part in enumerate(parts):
        if i % 2 == 1:  # a cue
            silence, volume, spd_mult = _cue_effect(part)
            volume = volume if volume is not None else 1.0
            spd = speed * spd_mult if spd_mult is not None else speed
            if silence:
                segments.append(np.zeros(int(silence * SAMPLE_RATE), dtype=np.float32))
        elif part.strip():
            segments.append(_synthesize(part.strip(), voice, spd) * volume)
    return np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)


def generate_tts_audio_tool_fn(text: str, speaker_one: str = None, speaker_two: str = None, language: str = "english",
                               voice_one: str = None, voice_two: str = None, speed: float = 1.0) -> str:
    """
    Generates TTS audio from text using the open-source Kokoro model (runs locally, CPU-friendly).
    Supports up to 2 speakers via 'Speaker: text' formatted lines.

    Args:
        text: The text to convert to speech. Use 'Speaker: text' format for multi-speaker.
        speaker_one: Optional name of the first speaker (e.g., 'Joe').
        speaker_two: Optional name of the second speaker (e.g., 'Jane').
        language: The target language of the text to read (default: English).
        voice_one: Kokoro voice id for the narrator / first speaker (default from config).
        voice_two: Kokoro voice id for the second speaker (default from config).
        speed: Speech speed multiplier (1.0 = normal).
    Returns:
        The path to the generated .wav file or an error message.
    """
    print(f"Generating TTS for: {text[:50]}...")
    v_one = voice_one or KOKORO_VOICE_SPEAKER_ONE
    v_two = voice_two or KOKORO_VOICE_SPEAKER_TWO

    try:
        voice_map = {}
        if speaker_one:
            voice_map[speaker_one] = v_one
        if speaker_two:
            voice_map[speaker_two] = v_two

        segments = []
        if voice_map:
            # Split "Speaker: text" lines and synthesize each with its own voice
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                match = re.match(r"^([^:]+):\s*(.+)$", line)
                if match and match.group(1) in voice_map:
                    speaker, spoken = match.groups()
                    segments.append(_synthesize_with_cues(spoken, voice_map[speaker], speed))
                else:
                    segments.append(_synthesize_with_cues(line, v_one, speed))
        else:
            segments.append(_synthesize_with_cues(text, v_one, speed))

        audio = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
        if audio.size == 0:
            return "Error: No audio generated."

        timestamp = int(time.time())
        filename = f"generated_audio_{timestamp}.wav"
        output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename

        sf.write(output_path, audio, 24000)
        print(f"TTS audio generated and saved to: {output_path}")
        return output_path

    except Exception as e:
        return f"An error occurred during TTS generation: {str(e)}"
