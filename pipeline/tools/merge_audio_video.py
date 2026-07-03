import subprocess
import json
import os

def get_duration(file_path):
    """Get duration of a file in seconds using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "quiet", 
        "-print_format", "json", 
        "-show_format", 
        "-show_streams", 
        file_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}")
    except (KeyError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to parse ffprobe output for {file_path}: {e}")

def merge_audio_video_tool_fn(video_path: str, audio_path: str, output_path: str = "output.mp4") -> str:
    """
    Merges audio and video files. The video is retimed (sped up or slowed down)
    so it ends exactly when the audio ends.
    
    Args:
        video_path: Path to the input video file.
        audio_path: Path to the input audio file.
        output_path: Path for the merged output file.
        
    Returns:
        Path to the output file if successful, or error message.
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found at {video_path}"
    if not os.path.exists(audio_path):
        return f"Error: Audio file not found at {audio_path}"
        
    try:
        video_dur = get_duration(video_path)
        audio_dur = get_duration(audio_path)
        
        print(f"Video duration: {video_dur:.2f}s")
        print(f"Audio duration: {audio_dur:.2f}s")
        
        # Retime the video so the drawing finishes when the narration ends,
        # clamped so it never looks unnaturally rushed or sluggish:
        # at most 3x faster, at most 2x slower. Any leftover gap is covered by
        # freezing the last frame (video short) or trailing silence (audio short).
        ratio = audio_dur / video_dur if video_dur > 0 else 1.0
        ratio = min(max(ratio, 1 / 3), 2.0)
        retimed_dur = video_dur * ratio
        freeze_pad = max(0.0, audio_dur - retimed_dur)
        filter_complex = (
            f"[0:v]setpts=PTS*{ratio:.6f},fps=25,"
            f"tpad=stop_mode=clone:stop_duration={freeze_pad:.3f}[v];"
            f"[1:a]apad[a]"
        )
        cmd = [
            "ffmpeg",
            "-y", # Overwrite output if exists
            "-i", video_path,
            "-i", audio_path,
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "[a]",
            "-pix_fmt", "yuv420p", # Ensure compatibility
            "-shortest", # apad is infinite, so output length = video length
            output_path,
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return f"Error during ffmpeg execution: {result.stderr}"
        else:
            print(f"Successfully created: {output_path}")
            return output_path
            
    except Exception as e:
        return f"Error in merge_audio_video_tool: {str(e)}"
