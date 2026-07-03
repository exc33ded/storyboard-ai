import os
import math
import time
import subprocess
import cv2
import numpy as np
from . import utils


def build_montage(image_paths, tile_wd=640, tile_ht=360, cols=None):
    """
    Tiles the scene images left-to-right, top-to-bottom on one white canvas.
    Returns (montage, last_tile_rect) where last_tile_rect is (x, y, w, h)
    of the final scene's tile — the zoom-out starts framed on it.
    """
    n = len(image_paths)
    cols = cols or int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    montage = np.full((rows * tile_ht, cols * tile_wd, 3), 255, dtype=np.uint8)

    last_rect = (0, 0, tile_wd, tile_ht)
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (tile_wd, tile_ht))
        r, c = divmod(i, cols)
        y, x = r * tile_ht, c * tile_wd
        montage[y : y + tile_ht, x : x + tile_wd] = img
        last_rect = (x, y, tile_wd, tile_ht)
    return montage, last_rect


def zoom_out_ending_tool_fn(
    image_paths: list,
    output_path: str = None,
    duration_sec: float = 4.0,
    hold_sec: float = 2.0,
    frame_rate: int = 25,
    out_wd: int = 1920,
    out_ht: int = 1080,
) -> str:
    """
    Renders the classic whiteboard-video ending: the camera starts framed on the
    last scene, then pulls back to reveal every scene tiled on one big canvas,
    and holds on the full view. Output has a silent audio track so it can be
    concatenated with the narrated scene videos.

    Args:
        image_paths: Final scene images, in scene order.
        output_path: Where to write the MP4. Defaults to the global output dir.
        duration_sec: Length of the zoom-out motion.
        hold_sec: How long to hold the full-canvas view at the end.
        frame_rate: Output FPS (match the scene videos, default 25).
        out_wd / out_ht: Output resolution.

    Returns:
        Path to the MP4, or an error message.
    """
    image_paths = [p for p in image_paths if p and os.path.exists(p)]
    if len(image_paths) < 2:
        return "Error: Need at least 2 scene images for a zoom-out ending."

    montage, (lx, ly, lw, lh) = build_montage(image_paths)
    mh, mw = montage.shape[:2]

    # Save the full canvas as a thumbnail (served by the webapp result card)
    if utils.GLOBAL_OUTPUT_DIR:
        cv2.imwrite(os.path.join(utils.GLOBAL_OUTPUT_DIR, "canvas_montage.png"), montage)

    # Start window: the last scene's tile expanded to output aspect ratio.
    aspect = out_wd / out_ht
    start_w = lw if lw / lh >= aspect else lh * aspect
    start_h = start_w / aspect
    start_cx, start_cy = lx + lw / 2, ly + lh / 2

    # End window: the whole montage, letterboxed to output aspect ratio.
    end_w = mw if mw / mh >= aspect else mh * aspect
    end_h = end_w / aspect
    end_cx, end_cy = mw / 2, mh / 2

    # Pad montage with white so crop windows never leave the image.
    pad = int(max(end_w - mw, end_h - mh, start_w, start_h) / 2) + 2
    padded = cv2.copyMakeBorder(montage, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    if output_path is None:
        output_path = os.path.join(utils.GLOBAL_OUTPUT_DIR or ".", f"zoom_out_ending_{int(time.time())}.mp4")
    temp_path = output_path + "_noaudio.mp4"

    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (out_wd, out_ht))
    n_frames = int(duration_sec * frame_rate)
    for f in range(n_frames):
        t = f / max(n_frames - 1, 1)
        t = t * t * (3 - 2 * t)  # smoothstep ease in/out
        w = start_w + (end_w - start_w) * t
        h = start_h + (end_h - start_h) * t
        cx = start_cx + (end_cx - start_cx) * t + pad
        cy = start_cy + (end_cy - start_cy) * t + pad
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        crop = padded[y0 : y0 + int(h), x0 : x0 + int(w)]
        writer.write(cv2.resize(crop, (out_wd, out_ht)))

    final_frame = cv2.resize(
        padded[int(end_cy - end_h / 2) + pad : int(end_cy + end_h / 2) + pad,
               int(end_cx - end_w / 2) + pad : int(end_cx + end_w / 2) + pad],
        (out_wd, out_ht),
    )
    for _ in range(int(hold_sec * frame_rate)):
        writer.write(final_frame)
    writer.release()

    # Re-encode with a silent audio track so scene concat (which maps audio) works.
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_path,
        "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-shortest",
        "-map", "0:v", "-map", "1:a",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        os.remove(temp_path)
    except OSError:
        pass
    if result.returncode != 0 or not os.path.exists(output_path):
        return f"Error adding silent audio to zoom-out ending: {result.stderr[-500:]}"
    return output_path
