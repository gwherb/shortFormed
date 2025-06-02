#!/usr/bin/env python3
"""
Simple Story-to-Video Pipeline Skeleton

A bare-bones implementation that we'll build up step by step.
"""

import os
import sys
import json
import random
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip
import math
# import whisper
import whisper_timestamped as whisper
from utils import *
import cv2
import subprocess

# We'll add imports as we implement each step
# import openai
# from elevenlabs import generate, set_api_key
# from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
# import whisper

load_dotenv()

class SimplePipeline:
    def __init__(self):
        self.config = self.load_config()
        self.video_dir = Path("minecraft_videos")
        self.output_dir = Path("output")
        self.music_dir = Path("background_music")
        self.output_dir.mkdir(exist_ok=True)
        self.openai_client = OpenAI(api_key=self.config["openai_api_key"])
        self.elabs_client = ElevenLabs(api_key=self.config["elevenlabs_api_key"])
        
    
    def load_config(self):
        """Load basic configuration"""
        config_file = "config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                return json.load(f)
        else:
            # Create basic config
            config = {
                "openai_api_key": "your_key_here",
                "elevenlabs_api_key": "your_key_here",
                "story_theme": "absurd office situations",
                "target_duration": 45
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("Created config.json - please update with your API keys")
            return config
    
    def step1_generate_story(self):
        """Step 1: Generate story using LLM"""
        print("Step 1: Generating story...")
        
        try:
            prompt = f"Generate a short story about {self.config['story_theme']} in 6-8 sentences. Only include the story without additional text such as a title. Make the story from the perspective of one of the characters. Make it entertaining, engaging, and suitable for a short video. Additionally, use language for a modern audience, avoid formal or corporate langueage."
            
            response = self.openai_client.responses.create(
                model='gpt-4.1',
                input=prompt
            )
            
            story = response.output_text
        except Exception as e:
            print(f"Error generating story: \n{e}")
            story = None
        
        if story:
            print(f"Generated story: {story[:50]}...")
        return story
    
    def step2_text_to_speech(self, story_text):
        """Step 2: Convert text to speech"""
        print("Step 2: Converting text to speech...")
        
        try:
            audio = self.elabs_client.text_to_speech.convert(
                text=story_text,
                voice_id="cgSgspJ2msm6clMCkdW9",
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = self.output_dir / f"story_audio_{timestamp}.mp3"
            
            with open(audio_file, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
        
        except Exception as e:
            print(f"Error generating speech audio: \n{e}")
            audio = None
        
        if audio:
            print(f"Audio file created: {audio_file}")
        return audio, audio_file
    
    def step3_add_music(self, audio_file):
        """Step 3: Add background music (optional)"""
        print("Step 3: Adding background music...")
        
        try:
            music_files = list(self.music_dir.glob("*.mp3")) + list(self.music_dir.glob("*.wav")) + list(self.music_dir.glob("*.m4a"))

            if not music_files:
                print("No background music found")
                return audio_file
            
            file_num = random.randint(0, len(music_files))
            music_file = music_files[0]
            
            speech = AudioSegment.from_file(self.output_dir / audio_file)
            music = AudioSegment.from_file(str(music_file))
            
            background_volume_reduction = 15 # dB to reduce
            quiet_music = music - background_volume_reduction
            
            speech_duration = len(speech)
            
            # Loop music if needed
            if len(quiet_music) < speech_duration:
                loops_needed = (speech_duration // len(quiet_music)) + 1
                quiet_music = quiet_music * loops_needed
            
            # Trim music to exact speech length
            quiet_music = quiet_music[:speech_duration]
            
            # fade music in and out
            fade_duration = min(3000, speech_duration // 10) # 3 seconds or 10% of speech duration
            quiet_music = quiet_music.fade_in(fade_duration).fade_out(fade_duration)
            
            # overlay speech on top of music
            final_audio = quiet_music.overlay(speech)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"speech_with_music_{timestamp}.mp3"
            
            final_audio.export(str(output_file), format='mp3', bitrate='192K')
            
            print(f"Succesfully added background music")
        
        except Exception as e:
            print(f"Error applying background music to speech: \n{e}")
            final_audio = None
            
        return str(final_audio)
    
    def step4_select_video(self, target_duration):
        """Step 4: Select video from catalog"""
        print("Step 4: Selecting video from catalog...")
        
        # Find video files
        video_files = list(self.video_dir.glob("*.mp4"))
        if not video_files:
            print("No video files found in minecraft_videos/ directory")
            return None
        
        # Get random video
        selected_video = random.choice(video_files)
        
        # Convert audio duration to seconds
        target_duration_s = target_duration / 1000
        
        # Get random clip from video
        video = VideoFileClip(str(selected_video))
        video_duration = video.duration
        clip_start = random.randint(0, math.ceil(video_duration-target_duration_s))
        clip = VideoFileClip(selected_video).subclipped(clip_start, clip_start+target_duration_s)
        
        # Save video as test
        output_file = "clip.mp4"
        clip.write_videofile(output_file)
        
        print(f"Selected video: {selected_video.name}")
        return output_file
    
    def step5_create_captions(self, audio_file):
        """Step 5: Generate captions"""
        print("Step 5: Creating captions...")
        
        # Use whisper model to generate subtitle file
        audio = whisper.load_audio(audio_file)
        model = whisper.load_model("base")
        result = whisper.transcribe(model, audio)
        
        word_level = create_word_level_subtitles(result, 'word_level.srt')
        phrase_level = create_phrase_level_subtitles(result, 'phrase_level.srt')
            
        print(f"Caption clip created")
        captions = {
            'word_level': word_level,
            'phrase_level': phrase_level
        }
        
        return captions
    
    def step6_merge_video(self, video_file, audio_file, captions, caption_mode='word'):
        """Step 6: Add captions to video using OpenCV
        
        Args:
            video_file: Path to input video
            audio_file: Path to audio (not used in this version)
            captions: Dict with 'word_level' and 'phrase_level' SRT file paths
            caption_mode: 'word' for word-level captions, 'phrase' for phrase-level captions
        """
        print(f"Step 6: Adding {caption_mode}-level captions to video")
        
        # Parse captions based on mode
        if caption_mode == 'word':
            with open(captions['word_level'], 'r', encoding='utf-8') as f:
                caption_content = f.read()
        else:  # phrase mode
            with open(captions['phrase_level'], 'r', encoding='utf-8') as f:
                caption_content = f.read()
        
        subtitles = parse_srt(caption_content)
        print(f"Loaded {len(subtitles)} caption segments")
        
        # Create output video with captions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        captioned_video = self.output_dir / f"captioned_video_{caption_mode}_{timestamp}.mp4"
        
        # Open video capture
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_file}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} fps, {total_frames} frames")
        
        # Setup high-quality video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(captioned_video), fourcc, fps, (width, height))
        
        # Check if writer opened successfully
        if not out.isOpened():
            raise ValueError("Could not open video writer")
        
        frame_number = 0
        ret, frame = cap.read()
        
        while ret:
            current_time = frame_number / fps
            
            # Find active caption
            active_caption = ""
            for caption in subtitles:
                if caption['start'] <= current_time <= caption['end']:
                    active_caption = caption['text']
                    break
            
            # Add caption to frame if active
            if active_caption:
                frame = add_text_to_frame(frame, active_caption, width, height, caption_mode)
            
            out.write(frame)
            
            # Progress indicator
            if frame_number % int(fps * 5) == 0:  # Every 5 seconds
                progress = (frame_number / total_frames) * 100
                print(f"Processing: {progress:.1f}% complete")
            
            frame_number += 1
            ret, frame = cap.read()
        
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Captioned video saved: {captioned_video}")
        return str(captioned_video)
    
    def step7_upload_youtube(self, video_file, story_text):
        """Step 7: Upload to YouTube"""
        print("Step 7: Uploading to YouTube...")
        
        # TODO: Implement YouTube API upload
        print(f"Would upload video: {video_file}")
        print(f"With title: {story_text[:30]}...")
        
        return "https://youtube.com/watch?v=placeholder"
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("🚀 Starting Simple Pipeline...")
        print("=" * 50)
        
        try:
            # Step 1: Generate story
            story = self.step1_generate_story()
            
            # Step 2: Text to speech
            audio_file = self.step2_text_to_speech(story)
            
            # Step 3: Add music
            audio_with_music = self.step3_add_music(audio_file)
            
            # Step 4: Select video
            video_file = self.step4_select_video(self.config["target_duration"])
            if not video_file:
                print("❌ No video found - add videos to minecraft_videos/ directory")
                return None
            
            # Step 5: Create captions
            captions = self.step5_create_captions(audio_with_music)
            
            # Step 6: Merge everything
            final_video = self.step6_merge_video(video_file, audio_with_music, captions)
            
            # Step 7: Upload
            youtube_url = self.step7_upload_youtube(final_video, story)
            
            print("=" * 50)
            print("✅ Pipeline completed!")
            print(f"Result: {youtube_url}")
            
            return youtube_url
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            return None

def main():
    """Test the pipeline"""
    pipeline = SimplePipeline()
    result = pipeline.run_pipeline()
    
    if result:
        print(f"\n🎉 Success! Video URL: {result}")
    else:
        print("\n💥 Pipeline failed. Check the steps above.")

if __name__ == "__main__":
    main()


# Test individual steps
def test_step(step_number):
    """Test individual pipeline steps"""
    pipeline = SimplePipeline()
    
    if step_number == 1:
        return pipeline.step1_generate_story()
    elif step_number == 2:
        story = pipeline.step1_generate_story()
        audio, audio_file = pipeline.step2_text_to_speech(story)
        
        if audio:
            play(audio)
        
        return audio
    elif step_number == 3:
        return pipeline.step3_add_music("story_audio_20250530_153827.mp3")
    elif step_number == 4:
        audio = AudioSegment.from_file(pipeline.output_dir / "speech_with_music_20250530_161201.mp3")
        duration = len(audio)
        return pipeline.step4_select_video(duration)
    elif step_number == 5:
        return pipeline.step5_create_captions("output/speech_with_music_20250530_161201.mp3")
    elif step_number == 6:
        audio_file = pipeline.output_dir / "speech_with_music_20250530_161201.mp3"
        clip_file = "clip.mp4"
        captions =  pipeline.step5_create_captions("output/speech_with_music_20250530_161201.mp3")
        return pipeline.step6_merge_video(clip_file, audio_file, captions)
    else:
        print("Invalid step number (1-7)")


