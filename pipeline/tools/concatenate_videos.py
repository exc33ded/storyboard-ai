import subprocess
import os

from .merge_audio_video import get_duration

FADE_SEC = 0.3


def concatenate_videos_tool_fn(video_paths: list, output_path: str = "concatenated_output.mp4") -> str:
    """
    Concatenates multiple video files (with audio) into a single video, with a
    short crossfade between each pair. Inputs are normalized to 1920x1080 @ 25fps
    and stereo 44.1kHz so mixed-resolution scenes concatenate cleanly.

    Args:
        video_paths: A list of paths to video files to concatenate. Must have at least 2 videos.
        output_path: Path for the concatenated output file.

    Returns:
        Path to the output file if successful, or an error message.
    """
    if not isinstance(video_paths, list) or len(video_paths) < 2:
        return "Error: video_paths must be a list with at least 2 video file paths."

    for i, vp in enumerate(video_paths):
        if not os.path.exists(vp):
            return f"Error: Video file not found at index {i}: {vp}"

    n = len(video_paths)
    try:
        durations = [get_duration(vp) for vp in video_paths]
    except Exception as e:
        return f"Error probing video durations: {e}"

    cmd = ["ffmpeg", "-y"]
    for vp in video_paths:
        cmd.extend(["-i", vp])

    # Normalize every input so xfade/acrossfade accept them
    parts = []
    for i in range(n):
        parts.append(
            f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            f"pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v{i}]"
        )
        parts.append(
            f"[{i}:a]aresample=44100,"
            f"aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}]"
        )

    # Chain crossfades: each xfade offset is where the fade starts on the
    # accumulated timeline (running total of durations minus overlap).
    v_prev, a_prev = "[v0]", "[a0]"
    offset = durations[0] - FADE_SEC
    for i in range(1, n):
        v_out = "[v]" if i == n - 1 else f"[vx{i}]"
        a_out = "[a]" if i == n - 1 else f"[ax{i}]"
        parts.append(
            f"{v_prev}[v{i}]xfade=transition=fade:duration={FADE_SEC}:offset={offset:.3f}{v_out}"
        )
        parts.append(f"{a_prev}[a{i}]acrossfade=d={FADE_SEC}{a_out}")
        v_prev, a_prev = v_out, a_out
        offset += durations[i] - FADE_SEC

    filter_complex = ";".join(parts)
    cmd.extend([
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "[a]",
        "-pix_fmt", "yuv420p",
        output_path,
    ])

    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            return f"Error during ffmpeg execution: {result.stderr}"
        else:
            print(f"Successfully concatenated {n} videos to: {output_path}")
            return output_path
    except Exception as e:
        return f"Error in concatenate_videos_tool: {str(e)}"
