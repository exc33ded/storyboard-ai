# Voice, Image-Model, and Narration Improvements — Design

Date: 2026-07-04. Approved by user in brainstorming session.

## 1. Voice settings side panel

- `static/index.html`: a voice-settings button next to the generate controls opens a slide-in side panel with:
  - Narrator voice dropdown and second-speaker voice dropdown, options grouped by gender/accent from Kokoro's catalog (American/British, female/male; includes deep options like `am_onyx`).
  - Speech speed slider, 0.7–1.3, default 1.0.
  - Choices persisted in `localStorage`.
- `webapp.py`: `GenerateRequest` gains `voice_one: str | None`, `voice_two: str | None`, `tts_speed: float = 1.0`; passed to `run_pipeline`.
- `pipeline.py`: `run_pipeline` gains the same three params, forwarded to `generate_tts_audio_tool_fn`.
- `tools/tts.py`: `generate_tts_audio_tool_fn` accepts `voice_one`, `voice_two`, `speed`; falls back to config defaults when unset. `speed` is passed to Kokoro's pipeline call.

Skipped: per-scene voice overrides, in-browser voice preview.

## 2. Gemini-first image generation

- `config.py`: reorder the `PUTER_IMAGE_MODELS` default so the latest Gemini image models come first (better text/spelling rendering than FLUX), FLUX last. Existing fallback chain (try each model, then Hugging Face) is unchanged.

## 3. Fix spoken "[pause]"

- Root cause: `narration_refiner.py` instructs the LLM to add `[pause]`/`[softly]` cues; Kokoro reads them aloud.
- Fix: remove that instruction; in `tts.py`, strip bracketed cues before synthesis — `[pause]` and similar become a comma so rhythm survives.

## 4. More natural narration

- Rewrite enhancement guidance in `narration_refiner.py`: write for the ear — contractions, mostly short sentences with varied rhythm, direct address ("you"), no flowery written-prose vocabulary, no stage directions or bracketed cues. Fact-preservation rules unchanged.

## Verification

- Unit check: call `generate_tts_audio_tool_fn("Hello [pause] world", ...)` — output audio contains no spoken "pause" (string-level check on the stripped text).
- Manual: run webapp, open voice panel, pick a male voice + slower speed, generate a short video; confirm voice/speed applied and no "pause" spoken.
