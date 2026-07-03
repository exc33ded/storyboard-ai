"""
Transcribe Audio Tool - Extracts subtitles from video/audio using local Whisper.
Returns timestamped transcription in JSON format for subtitle generation.
"""
import os
import json
from faster_whisper import WhisperModel
from config import WHISPER_MODEL_SIZE
from . import utils

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_audio_tool_fn(file_path: str, output_path: str = None) -> str:
    """
    Extracts subtitles from a video or audio file using local Whisper transcription.

    Args:
        file_path: Path to the input video or audio file.
        output_path: Optional path for the output JSON file. If not provided,
                    saves with '_subtitles.json' suffix in output directory.

    Returns:
        Path to the JSON file containing subtitles if successful, or error message.

    JSON Output Format:
        {
            "subtitles": [
                {"start": 0.0, "end": 2.5, "text": "Hello world"},
                {"start": 2.5, "end": 5.0, "text": "This is a test"}
            ]
        }
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    print(f"Transcribing file: {file_path}")

    try:
        model = _get_model()
        segments, _info = model.transcribe(file_path, vad_filter=True)

        subtitles = [
            {"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()}
            for seg in segments
        ]
        print(f"Extracted {len(subtitles)} subtitle segments.")

        result_data = {"subtitles": subtitles}

        if not output_path:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            filename = f"{base_name}_subtitles.json"
            output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else os.path.join(os.path.dirname(file_path), filename)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        print(f"Subtitles saved to: {output_path}")
        return output_path

    except Exception as e:
        return f"Error during transcription: {str(e)}"
