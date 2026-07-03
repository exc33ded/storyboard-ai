"""
Subtitle Tool - Adds text subtitles to images using Pillow.
Provides high-quality anti-aliased text with automatic word wrapping.
"""
import os
from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int = 24) -> ImageFont.FreeTypeFont:
    """Get a TrueType font with fallback options."""
    font_paths = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.Draw) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_width = bbox[2] - bbox[0]
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def add_subtitle_tool_fn(
    image_path: str,
    subtitle_text: str,
    output_path: str = None,
    font_size: int = 24,
    padding: int = 12,
    bg_opacity: int = 220
) -> str:
    """
    Adds a subtitle to an image at the bottom with automatic word wrapping.
    
    Args:
        image_path: Path to the input image file.
        subtitle_text: The text to add as a subtitle.
        output_path: Optional path for the output image. If not provided,
                     saves with '_subtitled' suffix.
        font_size: Size of the subtitle font (default: 24).
        padding: Padding around the text in pixels (default: 10).
        bg_opacity: Opacity of background box 0-255 (default: 160).
        
    Returns:
        Path to the output image if successful, or error message.
    """
    # Validate input
    if not os.path.exists(image_path):
        return f"Error: Image file not found at {image_path}"
    
    if not subtitle_text or not subtitle_text.strip():
        return "Error: Subtitle text is empty"
    
    try:
        # Load the background image
        background = Image.open(image_path).convert('RGBA')
        width, height = background.size
        
        # Create transparent overlay for subtitle
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Font settings
        font = _get_font(font_size)
        max_text_width = width - (padding * 4)
        
        # Wrap text to fit width
        lines = _wrap_text(subtitle_text, font, max_text_width, draw)
        
        # Calculate text block dimensions
        line_height = int(font_size * 1.3)  # 1.3 line spacing
        total_text_height = len(lines) * line_height
        
        # Position at bottom of frame
        box_y = height - total_text_height - (padding * 3)
        box_height = total_text_height + (padding * 2)
        
        # Draw semi-transparent background box (minimal border)
        draw.rectangle(
            [(padding, box_y), (width - padding, box_y + box_height)],
            fill=(0, 0, 0, bg_opacity)
        )
        
        # Draw each line of text (centered with crisp outline)
        y_offset = box_y + padding
        for line in lines:
            # Get text width for centering
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            x_pos = (width - text_width) // 2
            
            # Draw text with crisp stroke outline (instead of blurry shadow)
            draw.text(
                (x_pos, y_offset), 
                line, 
                fill=(255, 255, 255, 255),  # Pure white
                font=font,
                stroke_width=2,              # Crisp outline
                stroke_fill=(0, 0, 0, 255)   # Black stroke
            )
            y_offset += line_height
        
        # Composite overlay onto background
        result = Image.alpha_composite(background, overlay)
        result = result.convert('RGB')
        
        # Determine output path
        if not output_path:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_subtitled{ext}"
        
        result.save(output_path, quality=95)
        print(f"Subtitle added successfully: {output_path}")
        return output_path
        
    except Exception as e:
        return f"Error adding subtitle: {str(e)}"
