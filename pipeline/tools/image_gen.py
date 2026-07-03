import os
import time
from huggingface_hub import InferenceClient
from config import IMAGE_GEN_MODEL, VISION_MODEL, HF_API_KEY
from . import utils

_hf_client = InferenceClient(api_key=HF_API_KEY) if HF_API_KEY else None

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

    if not HF_API_KEY:
        return "Error: HF_API_KEY not configured."

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

    try:
        image = _hf_client.text_to_image(full_prompt, model=IMAGE_GEN_MODEL)
        utils.record_hf_call()

        timestamp = int(time.time())
        filename = f"generated_image_{timestamp}.png"
        output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR, filename) if utils.GLOBAL_OUTPUT_DIR else filename
        image.save(output_path)

        print(f"Image generated and saved to: {output_path}")
        return output_path
    except Exception as e:
        return f"An error occurred during image generation: {str(e)}"
