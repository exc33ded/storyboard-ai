from config import VISION_MODEL, TTS_WORDS_PER_MINUTE
from .utils import vision_chat, image_to_data_url, _save_to_run_folder
import os

def refine_narration_tool_fn(original_narration: str, image_path: str, video_duration: float = None, global_plan: dict = None, language: str = "english") -> str:
    """
    Enhances the Director's narration script — preserving ALL information.
    
    The Director's narration is the PRIMARY source of truth. This tool only
    refines it for pacing and flow. It does NOT replace content with image
    descriptions.
    
    Args:
        original_narration: The Director's narration script (PRIMARY CONTENT).
        image_path: Path to the generated image (for visual context ONLY).
        video_duration: Duration of the animation in seconds (for pacing).
        global_plan: The Director's global plan (for tone/persona consistency).
        language: The target language for the narration script (default: English).
    Returns:
        An enhanced narration string with ALL original information preserved.
    """
    if not os.path.exists(image_path):
        print(f"Warning: Image not found for narration refinement at {image_path}. Using original.")
        return original_narration

    # Calculate target word count based on video duration and the TTS voice's
    # measured speaking pace (config.TTS_WORDS_PER_MINUTE)
    duration_guidance = ""
    if video_duration:
        target_words = int((video_duration / 60) * TTS_WORDS_PER_MINUTE)
        # Allow narration to be somewhat longer than video (audio can extend the final frame)
        max_words = int(target_words * 1.5)
        # Only bypass tightening if narration is significantly longer (2x the target) —
        # this happens when the Director deliberately writes a rich, long scene narration
        bypass_threshold = int(target_words * 2.0)
        original_words = len(original_narration.split())
        
        if original_words > bypass_threshold:
            duration_guidance = f"""
    PACING CONSTRAINT:
    - The animation is {video_duration:.1f} seconds long.
    - The Director's narration is intentionally rich and detailed ({original_words} words).
    - Do NOT shorten, compress, or tighten the narration. Preserve it in full.
    - The video will hold on the last frame to accommodate the longer audio. That is expected and fine.
    """
        else:
            duration_guidance = f"""
    PACING CONSTRAINT:
    - The animation is {video_duration:.1f} seconds long.
    - At the voice's speaking pace (~{TTS_WORDS_PER_MINUTE} words/min), aim for approximately {target_words}-{max_words} words.
    - If the Director's narration is already close to this length, make minimal changes.
    - If it's too short, EXPAND with more vivid storytelling detail (not new information).
    - If it's too long, TIGHTEN the language (but keep ALL facts/information).
    """

    persona = global_plan.get("narrative_persona", "Professional Storyteller") if global_plan else "Professional Storyteller"
    tone = global_plan.get("tone", "dramatic") if global_plan else "dramatic"
    narrative_arc = global_plan.get("narrative_arc", "") if global_plan else ""

    arc_context = ""
    if narrative_arc:
        arc_context = f"\n    Overall Story Arc: {narrative_arc}"

    prompt = f"""
    You are a Narration Enhancer working under the direction of a {persona}.
    
    DIRECTOR'S ORIGINAL NARRATION (THIS IS YOUR PRIMARY INPUT — PRESERVE IT):
    "{original_narration}"
    
    Tone: {tone}
    {arc_context}
    {duration_guidance}
    
    An image of the whiteboard animation frame is attached for visual context.
    
    YOUR TASK: Enhance the Director's narration for spoken delivery.
    
    ABSOLUTE RULES — VIOLATION IS FAILURE:
    1. PRESERVE ALL INFORMATION: Every fact, name, number, and detail from the Director's 
       narration MUST appear in your output. You are ENHANCING, not replacing.
    2. DO NOT DESCRIBE THE IMAGE: You must NEVER say things like "we see a drawing of..." 
       or "the whiteboard shows..." or "lines and shapes depict...". The narration is for 
       the AUDIENCE watching the video, not someone looking at a whiteboard.
    3. TELL THE STORY: The narration should sound like a compelling story being told to 
       an audience. Use the {tone} tone throughout.
    4. KEEP THE VOICE: Maintain the {persona} voice consistently.
    5. WRITE FOR THE EAR, NOT THE PAGE: This text goes straight into a text-to-speech
       engine that reads every character literally. Write it exactly as it should be
       spoken — plain sentences and punctuation only. NEVER include stage directions,
       bracketed cues, or delivery notes like [pause], [softly], (beat), or asterisks;
       the voice will read them out loud. To create a pause, end the sentence or use
       a comma or ellipsis.
    6. SOUND LIKE A PERSON TALKING: Use contractions (it's, don't, that's). Keep most
       sentences short, and vary the rhythm — a longer sentence, then a punchy one.
       Address the listener directly as "you" where it fits. Prefer everyday words
       over flowery written-prose vocabulary (no "tapestry", "testament", "delve",
       "myriad"). It should read like a great documentary narrator speaking, not an
       essay being read aloud.
    7. OUTPUT LANGUAGE: The refined narration MUST be written in {language}.

    WHAT YOU MAY DO:
    - Improve word choice for more vivid, engaging storytelling
    - Adjust sentence rhythm for better spoken delivery
    - Add transitional phrases for smoother flow
    - Expand briefly if the narration needs to be longer for the video duration
    - Add emotional coloring that matches the scene's mood

    WHAT YOU MUST NEVER DO:
    - Remove or replace factual content from the Director's narration
    - Add information that wasn't in the original narration
    - Describe what's drawn in the whiteboard image
    - Add meta-commentary about the video or animation
    - Change the core message or meaning
    - Include anything in brackets or parentheses that isn't meant to be spoken
    
    Output: Return ONLY the enhanced narration text. No explanations, no labels, no quotes.
    """

    try:
        response = vision_chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }],
        )

        result = response.choices[0].message.content.strip()
        # Clean any wrapping quotes the model might add
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1]
        
        _save_to_run_folder(f"Original: {original_narration}\nRefined: {result}\n---\n", "narration_refinement_log.txt", mode="a")
        return result
        
    except Exception as e:
        print(f"Error refining narration: {e}")
        return original_narration
