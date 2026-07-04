import os
import time
import urllib.parse

import requests
from huggingface_hub import InferenceClient
from config import IMAGE_GEN_MODEL, VISION_MODEL, HF_API_KEY, PUTER_AUTH_TOKEN, PUTER_IMAGE_MODELS
from . import utils

_hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None


def _puter_gen(prompt: str, output_path: str) -> str:
    """
    Generate an image via Puter's drivers API — the same POST /drivers/call the
    official puter.js SDK makes (driver 'ai-image', method 'generate').
    Tries each model in PUTER_IMAGE_MODELS until one succeeds.
    """
    errors = []
    for model in [m.strip() for m in PUTER_IMAGE_MODELS.split(",") if m.strip()]:
        try:
            r = requests.post(
                "https://api.puter.com/drivers/call",
                headers={
                    "Authorization": f"Bearer {PUTER_AUTH_TOKEN}",
                    "Content-Type": "text/plain;actually=json",
                },
                json={
                    "interface": "puter-image-generation",
                    "driver": "ai-image",
                    "method": "generate",
                    "args": {"prompt": prompt, "model": model, "ratio": {"w": 16, "h": 9}},
                },
                timeout=300,
            )
            r.raise_for_status()
            if r.headers.get("content-type", "").startswith("image/"):
                data = r.content
            else:
                body = r.json()
                if not body.get("success"):
                    raise RuntimeError(f"Puter error: {str(body)[:200]}")
                result = body["result"]
                if result.startswith("data:image"):
                    import base64
                    data = base64.b64decode(result.split(",", 1)[1])
                elif result.startswith("http"):
                    data = requests.get(result, timeout=120).content
                else:
                    raise RuntimeError(f"unexpected result: {result[:200]}")
            with open(output_path, "wb") as f:
                f.write(data)
            print(f"Puter image generated with {model}")
            return output_path
        except Exception as e:
            errors.append(f"{model}: {e}")
            print(f"  [!] Puter model {model} failed, trying next...")
    raise RuntimeError("all Puter models failed: " + " | ".join(errors)[:400])

# ponytail: no web-search grounding equivalent wired up here (dropped Google Search
# tool with the Gemini swap) — verification is vision-only now. Add a search step
# if false positives on ambiguous subjects become a problem.
def _verify_image_makes_sense(image_path: str, context_prompt: str) -> bool:
    """Verifies using the vision model if the given downloaded image makes sense for the prompt."""
    try:
        if not os.path.exists(image_path):
            return False

        prompt = f"Look at this image. Does this image genuinely represent the subject needed for this whiteboard animation prompt: '{context_prompt}'? Answer ONLY with 'YES' or 'NO'."

        response = utils.vision_chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": utils.image_to_data_url(image_path)}},
                ],
            }],
        )
        answer = response.choices[0].message.content.strip().upper()
        print(f"Image verification for {image_path}: {answer}")
        return "YES" in answer
    except Exception as e:
        print(f"Warning: Image verification failed: {e}")
        return False

def image_gen_tool_fn(prompt: str, reference_image_path: str = None, subject_reference_image_path: str = None) -> str:
    """
    Generates a whiteboard animation image using FLUX.2 [klein] 4B via the
    Hugging Face Inference API.

    Args:
        prompt: The specific text prompt to generate an image for.
        reference_image_path: Optional path to a previously generated image for aesthetic consistency.
        subject_reference_image_path: Optional path to a real-world photo of the subject.
    Returns:
        The path to the generated image or an error message.
    """

    full_prompt = prompt + " Ensure the generated image is in 16:9 aspect ratio (1920x1080). CRITICAL: DO NOT draw any hands, human arms, markers, pens, or people drawing. Draw ONLY the pure artwork on the whiteboard."

    # ponytail: HF's serverless inference endpoint takes text-only input, so
    # reference images can't be passed as conditioning like the old Gemini
    # multi-image call did — fold them into the prompt as style guidance
    # instead. Swap to a self-hosted FLUX endpoint if true image conditioning
    # (character/style consistency across scenes) is needed.
    if reference_image_path and os.path.exists(reference_image_path):
        full_prompt += " Maintain the same whiteboard line-art style, line weight, and color palette as the previous scene in this series."

    if subject_reference_image_path and os.path.exists(subject_reference_image_path):
        if _verify_image_makes_sense(subject_reference_image_path, prompt):
            full_prompt += " Keep the real-world subject's recognizable structural features accurate to reference photos."
        else:
            print(f"Skipping subject reference {subject_reference_image_path} as it was deemed not useful.")

    timestamp = int(time.time())
    filename = f"generated_image_{timestamp}.png"
    output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename

    if not PUTER_AUTH_TOKEN and not _hf_client:
        return "Error: set PUTER_AUTH_TOKEN or HF_API_KEY for image generation."

    if PUTER_AUTH_TOKEN:
        try:
            _puter_gen(full_prompt, output_path)
            print(f"Image generated via Puter and saved to: {output_path}")
            return output_path
        except Exception as e:
            if not _hf_client:
                return f"An error occurred during image generation: {e}"
            print(f"  [!] Puter image gen failed ({e}) — falling back to Hugging Face.")

    try:
        image = _hf_client.text_to_image(full_prompt, model=IMAGE_GEN_MODEL)
        utils.record_hf_call()
        image.save(output_path)
        print(f"Image generated and saved to: {output_path}")
        return output_path
    except Exception as e:
        return f"An error occurred during image generation: {str(e)}"
