import cv2
from moviepy import TextClip
import re
from pydantic import BaseModel

def create_word_level_subtitles(whisper_result, output_file="subtitles.srt"):
    """
    Create SRT subtitle file with one word appearing at a time
    
    Args:
        whisper_result: The result object from whisper-timestamped
        output_file: Path to save the SRT file
    
    Returns:
        str: Path to the created subtitle file
    """
    
    output_file = str(output_file)
    
    def format_timestamp(seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    subtitle_lines = []
    subtitle_index = 1
    
    # Extract all words from all segments
    for segment in whisper_result.get("segments", []):
        words = segment.get("words", [])
        
        for word in words:
            word_text = re.sub(r"[^\w\s']", "", word["text"].strip())
            if not word_text:  # Skip empty words
                continue
                
            start_time = format_timestamp(word["start"])
            end_time = format_timestamp(word["end"])
            
            # Create SRT entry for this word
            subtitle_lines.append(f"{subtitle_index}")
            subtitle_lines.append(f"{start_time} --> {end_time}")
            subtitle_lines.append(word_text)
            subtitle_lines.append("")  # Empty line between entries
            
            subtitle_index += 1
    
    # Write to file
    subtitle_content = "\n".join(subtitle_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(subtitle_content)
    
    print(f"Created word-level subtitle file: {output_file}")
    print(f"Total words: {subtitle_index - 1}")
    
    return output_file

def create_phrase_level_subtitles(whisper_result, output_file="phrase_subtitles.srt", phrase_timing=0.2):
    """
    Create SRT subtitle file with multiple words per subtitle
    
    Args:
        whisper_result: The result object from whisper-timestamped
        output_file: Path to save the SRT file
        words_per_subtitle: Number of words to group together
    
    Returns:
        str: Path to the created subtitle file
    """
    
    output_file = str(output_file)
    
    def format_timestamp(seconds):
        """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    subtitle_lines = []
    subtitle_index = 1
    
    # Extract all words from all segments
    all_words = []
    for segment in whisper_result.get("segments", []):
        words = segment.get("words", [])
        for word in words:
            if word["text"].strip():  # Skip empty words
                all_words.append(word)
                    
    # Group words into phrases
    i = 0
    while i < len(all_words):
        word_group = [word for word in all_words[i:] if word["start"] - all_words[i]["start"] < phrase_timing]
        i = i + len(word_group)
                
        if not word_group:
            continue
            
        # Get start time from first word, end time from last word
        start_time = format_timestamp(word_group[0]["start"])
        end_time = format_timestamp(word_group[-1]["end"])
        
        # Combine word texts
        phrase_text = " ".join(re.sub(r"[^\w\s'-]", "", word["text"].strip()) for word in word_group)
        
        # Create SRT entry for this phrase
        subtitle_lines.append(f"{subtitle_index}")
        subtitle_lines.append(f"{start_time} --> {end_time}")
        subtitle_lines.append(phrase_text)
        subtitle_lines.append("")  # Empty line between entries
        
        subtitle_index += 1
    
    # Write to file
    subtitle_content = "\n".join(subtitle_lines)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(subtitle_content)
    
    print(f"Created phrase-level subtitle file: {output_file}")
    print(f"Total phrases: {subtitle_index - 1}")
    
    return output_file

def parse_srt_file(srt_content):
    """Parse SRT content and return list of subtitle dictionaries"""
    import re
    
    # If srt_content is a file path, read it
    if isinstance(srt_content, str) and srt_content.endswith('.srt'):
        with open(srt_content, 'r', encoding='utf-8') as f:
            content = f.read()
    else:
        content = srt_content
    
    # Parse SRT format
    pattern = r'(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\d+\s*\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    subtitles = []
    for match in matches:
        start_time = srt_time_to_seconds(match[1])
        end_time = srt_time_to_seconds(match[2])
        text = match[3].strip()
        
        subtitles.append({
            'text': text,
            'start': start_time,
            'end': end_time
        })
    
    return subtitles

def srt_time_to_seconds(time_str):
    """Convert SRT time format to seconds"""
    h, m, s_ms = time_str.split(':')
    s, ms = s_ms.split(',')
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

def srt_time_to_seconds(time_str):
    """Convert SRT time format to seconds"""
    # Format: "00:00:01,500" -> 1.5 seconds
    time_part, ms_part = time_str.split(',')
    h, m, s = map(int, time_part.split(':'))
    ms = int(ms_part)
    return h * 3600 + m * 60 + s + ms / 1000.0

def add_text_to_frame(frame, text, width, height, caption_mode='phrase'):
    """Add text overlay to frame centered on screen"""
    if not text.strip():
        return frame
    
    # High-def font settings
    font = cv2.FONT_HERSHEY_DUPLEX
    
    # Scale font based on video resolution for hi-def
    base_scale = min(width, height) / 1000.0  # Scale factor based on resolution
    
    if caption_mode == 'word':
        font_scale = 1.8 * base_scale
        thickness = max(2, int(3 * base_scale))
        color = (255, 255, 255)  # White text
        outline_color = (0, 0, 0)  # Black outline
        outline_thickness = thickness + 2
    else:  # phrase mode
        font_scale = 2.2 * base_scale
        thickness = max(3, int(4 * base_scale))
        color = (255, 255, 255)  # White text
        outline_color = (0, 0, 0)  # Black outline
        outline_thickness = thickness + 2
    
    # Split long text into multiple lines
    words = text.split()
    lines = []
    current_line = ""
    max_width = int(width * 0.85)  # Use 85% of screen width
    
    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_size = cv2.getTextSize(test_line, font, font_scale, thickness)[0]
        
        if text_size[0] <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    # Calculate positioning for center of screen
    line_height = int(font_scale * 50)
    total_text_height = len(lines) * line_height
    center_y = height // 2
    start_y = center_y - (total_text_height // 2) + line_height
    
    # Draw each line with outline for better visibility
    for i, line in enumerate(lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = start_y + i * line_height
        
        # Add semi-transparent background rectangle
        padding = int(15 * base_scale)
        bg_alpha = 0.7
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (text_x - padding, text_y - text_size[1] - padding),
                     (text_x + text_size[0] + padding, text_y + padding),
                     (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 1 - bg_alpha, overlay, bg_alpha, 0)
        
        # Add text outline for better readability
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, outline_color, outline_thickness)
        # Add main text
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, color, thickness)
    
    return frame

def create_custom_subtitles(self, subtitle_data):
    """Create a list of TextClips with custom effects"""
    text_clips = []
    
    for i, subtitle in enumerate(subtitle_data):
        # Alternating colors
        color = 'lime' if i % 2 == 0 else 'cyan'
        
        # You can add different effects based on position, timing, etc.
        if i % 5 == 0:  # Every 5th word gets special treatment
            font_size = 100
            position = ('center', 200)  # Higher up
        else:
            font_size = 80
            position = 'center'
        
        text_clip = TextClip(
            text=subtitle['text'],
            font='KOMIKAX_.ttf',
            font_size=font_size,
            color=color,
            stroke_color='black',
            stroke_width=15
        ).with_position(position).with_start(subtitle['start']).with_end(subtitle['end'])
        
        # Add fade in/out effects
        text_clip = text_clip.crossfadein(0.1).crossfadeout(0.1)
        
        text_clips.append(text_clip)
    
    return text_clips

class StructuredResponse(BaseModel):
    story: str
    title: str
    description: str
    tags: list[str]
    gender: str
