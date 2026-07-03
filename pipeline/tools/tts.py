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


def _synthesize(text: str, voice: str) -> np.ndarray:
    pipeline = _get_pipeline()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)


def generate_tts_audio_tool_fn(text: str, speaker_one: str = None, speaker_two: str = None, language: str = "english") -> str:
    """
    Generates TTS audio from text using the open-source Kokoro model (runs locally, CPU-friendly).
    Supports up to 2 speakers via 'Speaker: text' formatted lines.

    Args:
        text: The text to convert to speech. Use 'Speaker: text' format for multi-speaker.
        speaker_one: Optional name of the first speaker (e.g., 'Joe').
        speaker_two: Optional name of the second speaker (e.g., 'Jane').
        language: The target language of the text to read (default: English).
    Returns:
        The path to the generated .wav file or an error message.
    """
    print(f"Generating TTS for: {text[:50]}...")

    try:
        voice_map = {}
        if speaker_one:
            voice_map[speaker_one] = KOKORO_VOICE_SPEAKER_ONE
        if speaker_two:
            voice_map[speaker_two] = KOKORO_VOICE_SPEAKER_TWO

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
                    segments.append(_synthesize(spoken, voice_map[speaker]))
                else:
                    segments.append(_synthesize(line, KOKORO_VOICE_SPEAKER_ONE))
        else:
            segments.append(_synthesize(text, KOKORO_VOICE_SPEAKER_ONE))

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
