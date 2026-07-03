import json
import re
from typing import List, Dict, Any
from config import TEXT_MODEL, TEXT_EXTRA_BODY, TTS_WORDS_PER_MINUTE
from .utils import text_client, _save_to_run_folder

def director_tool_fn(user_instructions: str, research_material: str = None, language: str = "english", enable_veo: bool = False, veo_direction_by_director: bool = False, target_duration_minutes: float = None) -> Dict[str, Any]:
    """
    Acts as the Video Director & Writer — plans the entire video journey.

    The Director decides how many scenes are needed, writes the narration script
    for each scene (as a storyteller, not just descriptions), plans the visual
    setup, and defines the overall narrative arc.

    Args:
        user_instructions: The user's original topic/instructions.
        research_material: Optional detailed research report to incorporate.
        language: The target language for the narration script (default: English).
        enable_veo: Whether Veo video generation is enabled.
        veo_direction_by_director: Whether the director should explicitly draft Veo prompts.
        target_duration_minutes: Optional target total video length in minutes.
    Returns:
        A dictionary with 'global_plan' and 'scenes'.
    """
    
    # If research material is provided, include it; otherwise just use instructions
    research_block = ""
    if research_material and research_material != user_instructions:
        research_block = f"""
    
    Deep Research Material (USE THIS — it contains rich, detailed information that MUST be woven into your narration):
    ---
    {research_material}
    ---
    """

    duration_block = ""
    if target_duration_minutes:
        target_seconds = target_duration_minutes * 60
        target_words = int(target_seconds / 60 * TTS_WORDS_PER_MINUTE)
        approx_scenes = max(1, round(target_seconds / 35))
        words_per_scene = max(30, round(target_words / approx_scenes / 10) * 10)
        duration_block = f"""
    TARGET VIDEO LENGTH: The user wants a total video of approximately {target_duration_minutes:.1f} minute(s)
    (~{target_seconds:.0f} seconds, ~{target_words} words of narration total at natural speaking pace).
    Plan roughly {approx_scenes} scene(s), and — CRITICAL — write approximately {words_per_scene} WORDS OF
    NARRATION PER SCENE. A one- or two-sentence narration is far too short and will make the video much
    shorter than the user asked for. Each scene's narration should be a full spoken paragraph.
    Do not pad with filler; if the topic genuinely needs fewer/more scenes, prioritize a coherent story,
    but keep the per-scene narration length so the total speaking time still lands near the target.
    """

    veo_instruction = ""
    veo_schema_field = ""
    if enable_veo and veo_direction_by_director:
        veo_instruction = (
            "- 'veo_prompt': Write a descriptive prompt for Veo video generation. This prompt should describe "
            "how the elements in the whiteboard drawing should come to life, animate, move, or transition. "
            "The video starts from the final whiteboard drawing, so describe the motion continuation. "
            "Keep it focused on movement, actions, and style continuity from the sketch. Avoid adding marker pens/hands."
        )
        veo_schema_field = '\n          "veo_prompt": "...",'

    prompt = f"""
    You are an award-winning Video Director, Writer, and Storyteller.
    You are planning a whiteboard animation video. Your job is to craft the ENTIRE video — 
    the narrative arc, the script, and the visual direction for every single scene.
    
    User's Topic / Instructions:
    "{user_instructions}"
    {research_block}
    {duration_block}

    YOUR TASK — Plan the complete video:
    
    STEP 1: Analyze the topic and decide:
    - What TONE fits? (informative, dramatic, playful, sad, etc.)
    - What is the NARRATIVE ARC? (beginning hook → build-up → climax → resolution)
    - Who is narrating? (a professional explainer, a storyteller, a historian, etc.)
    - How many scenes are needed? (CRITICAL: Follow these rules strictly)
    - The goal is that no single scene should have more than ~30-40 seconds of narration.
    
    STEP 2: For EACH scene, you must provide:
    - 'scene_number': Sequential number
    - 'summary': A 1-line summary of what this scene accomplishes in the narrative arc
    - 'narration': The FULL spoken script for this scene. THIS IS THE MOST IMPORTANT PART.
    - 'description': Visual description for the image generator (what should be DRAWN in this frame)
    - 'visual_setup': Specific visual direction for this frame (composition, key elements, focal points)
    {veo_instruction}
    - 'search_query': (OPTIONAL) If this scene features a specific real-world person, historical figure, or landmark, provide a search query.
    - 'text_overlay': (OPTIONAL) If you want specific impactful text visually rendered.
    - 'key_information': Any critical facts/data from the research that this scene must convey
    - 'emotional_beat': The emotional tone of this specific scene
    
    CRITICAL RULES:
    - LANGUAGE: The entire script's narration and summary values MUST be written in {language}.
    - ATTRACTIVE PACING & TONE: You MUST detect pacing instructions from the user.
    
    Output Format (Strict JSON) where all values (specifically 'narration' and 'summary') are in the language '{language}', but the JSON keys remain exactly as defined below in English:
    {{
      "global_plan": {{
        "title": "Video title",
        "tone": "informative" | "dramatic" | "educational" | "cautionary",
        "narrative_persona": "e.g., Wise Storyteller",
        "visual_style": "e.g., Clean Whiteboard Animation",
        "pacing": "e.g., steady/educational",
        "narrative_arc": "...",
        "target_audience": "...",
        "total_scenes": <number>
      }},
      "scenes": [
        {{
          "scene_number": 1,
          "summary": "...",
          "narration": "...",
          "description": "...",
          "visual_setup": "...",{veo_schema_field}
          "search_query": "...",
          "text_overlay": "...",
          "key_information": "...",
          "emotional_beat": "..."
        }},
        ...
      ]
    }}
    """
    
    try:
        response = text_client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            extra_body=TEXT_EXTRA_BODY,
        )
        result = json.loads(response.choices[0].message.content)
        _save_to_run_folder(json.dumps(result, indent=2), "video_plan.json")
        return result
    except Exception as e:
        print(f"Error in director_tool: {e}")
        # Fallback to a basic structure if parsing fails
        return {
            "global_plan": {
                "title": "Untitled Video",
                "tone": "informative", 
                "narrative_persona": "Professional Storyteller", 
                "visual_style": "Clean Whiteboard Animation", 
                "pacing": "steady",
                "narrative_arc": "Linear exploration of the topic",
                "target_audience": "general public",
                "total_scenes": 1
            },
            "scenes": [{
                "scene_number": 1,
                "summary": "Error parsing",
                "description": "Error parsing", 
                "narration": "Error parsing", 
                "visual_setup": "Simple sketch",
                "search_query": "",
                "text_overlay": "",
                "key_information": "",
                "emotional_beat": "neutral"
            }]
        }
