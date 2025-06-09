#!/usr/bin/env python3
"""
Simple Story-to-Video Pipeline Skeleton

A bare-bones implementation that we'll build up step by step.
"""

import os
import sys
import shutil
import json
import random
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip, ImageClip
from moviepy.video.fx.Crop import Crop
import math
import whisper_timestamped as whisper
from youtube_upload.client import YoutubeUploader
from utils import *


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
        self.short_output_dir = Path("short_videos")
        self.short_output_dir.mkdir(exist_ok=True)
        self.long_output_dir = Path("long_videos")
        self.long_output_dir.mkdir(exist_ok=True)
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        self.music_dir = Path("background_music")
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
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print("Created config.json - please update with your API keys")
            return config
    
    def step1_generate_story(self):
        """Step 1: Generate story using LLM"""
        print("Step 1: Generating story...")
        
        try:
            with open("prompt.txt", "r") as f:
                prompt = f.read()
            
            prompt = prompt.format(theme = self.config["story_theme"])
            
            client = self.openai_client
            
            response = client.responses.parse(
                        model="gpt-4o-2024-08-06",
                        input=[
                            {
                                "role": "system",
                                "content": "You are a helpful assistant that generates creative stories.",
                            },
                            {
                                "role": "user",
                                "content": prompt  # Use the formatted prompt
                            },
                        ],
                        text_format=StructuredResponse,
                    )
            
            results = response.output_parsed
            
            story = results.story
            title = results.title
            description = results.description
            tags = results.tags
            gender = results.gender
            
            with open(self.temp_dir / "story.txt", "x") as f:
                f.write(story)
            
        except Exception as e:
            print(f"Error generating story: \n{e}")
            return None
        
        return story, title, description, tags, gender
    
    def stepX_generate_reddit_image(self, title_text):
        """Step X: Generate Image to overlay at the start of the video"""
        print("Step X: Generating Reddit Image...")
        
        title_text = title_text.replace("|", "\n")
        
        title_len = len(title_text)
        size_correction = (40 - title_len)/25 if title_len < 40 else 1
        
        image = ImageClip("reddit_template.png")
        
        title_overlay = TextClip(
            text=title_text,
            size=(int(0.8 * size_correction * image.size[0]), int(0.8 * size_correction * image.size[1])),
            font="impact.ttf",
            color="black",
            margin=(200,200)
        ).with_position(('center', 'center'))
        
        username_overlay = TextClip(
            text="Anonymous",
            size=(int(0.2 * image.size[0]), int(0.2 * image.size[1])),
            font="impact.ttf",
            color="black",
            margin=(200,200)
        ).with_position((135, -115)) #(Horizontal Shift, Vertical Shift)
        
        final_image = CompositeVideoClip([image, title_overlay, username_overlay])
        
        image_output = self.temp_dir / "overlay.png"
        
        final_image.save_frame(image_output)
        
        return image_output
    
    def step2_text_to_speech(self, story, title, gender):
        """Step 2: Convert text to speech"""
        print("Step 2: Converting text to speech...")
        
        voice_id = 'onwK4e9ZLuTAKqWW03F9' if gender == 'M' else 'cgSgspJ2msm6clMCkdW9'
        
        try:
            audio = self.elabs_client.text_to_speech.convert(
                text=title.replace("|", " ") + "\n\n" + story,
                voice_id=voice_id,
                model_id="eleven_multilingual_v2",
                output_format="mp3_44100_128"
            )

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_file = self.temp_dir / f"story_audio_{timestamp}.mp3"
            
            with open(audio_file, 'wb') as f:
                for chunk in audio:
                    f.write(chunk)
        
        except Exception as e:
            print(f"Error generating speech audio: \n{e}")
            audio = None
            duration = None
        
        if audio:
            print(f"Audio file created: {audio_file}")
            duration = len(AudioSegment.from_file(audio_file))
            
        return audio, audio_file, duration
    
    def step3_add_music(self, audio_file, duration):
        """Step 3: Add background music (optional)"""
        print("Step 3: Adding background music...")
        
        try:
            music_files = list(self.music_dir.glob("*.mp3")) + list(self.music_dir.glob("*.wav")) + list(self.music_dir.glob("*.m4a"))

            if not music_files:
                print("No background music found")
                return audio_file
            
            file_num = random.randint(0, len(music_files)-1)
            music_file = music_files[file_num]
            
            speech = AudioSegment.from_file(audio_file)
            music = AudioSegment.from_file(str(music_file))
            
            background_volume_reduction = 15 # dB to reduce
            quiet_music = music - background_volume_reduction
            
            speech_duration = duration
            
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
            output_file = self.temp_dir / f"speech_with_music_{timestamp}.mp3"
            
            final_audio.export(str(output_file), format='mp3', bitrate='192K')
            
            print(f"Succesfully added background music")
        
        except Exception as e:
            print(f"Error applying background music to speech: \n{e}")
            output_file = None
            
        return output_file
    
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
        output_file = self.temp_dir / "clip.mp4"
        clip.write_videofile(output_file)
        
        print(f"Selected video: {selected_video.name}")
        return output_file
    
    def step5_create_captions(self, audio_file):
        """Step 5: Generate captions"""
        print("Step 5: Creating captions...")
        
        # Use whisper model to generate subtitle file
        audio = whisper.load_audio(audio_file)
        model = whisper.load_model("small")
        result = whisper.transcribe(model, audio)
        
        word_level = create_word_level_subtitles(result, 'word_level.srt')
        phrase_level = create_phrase_level_subtitles(result, 'phrase_level.srt')
            
        print(f"Caption clip created")
        captions = {
            'word_level': word_level,
            'phrase_level': phrase_level
        }
        
        return captions
    
    def step6_merge_video(self, video_file, audio_file, captions, reddit_image, caption_mode='phrase_level'):
        """Step 6: Create custom subtitles with individual TextClips"""
        print(f"Step 6: Adding custom {caption_mode}-level captions to video")
        
        # Load in image overlay
        reddit_image = ImageClip(reddit_image)
        reddit_image = reddit_image.resized(1/4)
        reddit_image = reddit_image.with_position(('center', 'center'))
        reddit_image = reddit_image.with_duration(2.5)
        
        # Load video and audio
        video = VideoFileClip(video_file)
        audio = AudioFileClip(audio_file)
        video = video.with_audio(audio)
        
        # Parse your SRT file to get timing data
        # Assuming you have a function that returns: 
        # [{'text': 'word', 'start': 1.5, 'end': 2.0}, ...]
        subtitle_data = parse_srt_file(captions[caption_mode])
        
        # Create individual TextClips for each word/phrase
        text_clips = []
        word_counter = 0
        
        for subtitle in subtitle_data:
            color = 'lime' if word_counter % 2 == 0 else 'white'
            word_counter += 1
            
            text_clip = TextClip(
                text=subtitle['text'],
                font='KOMIKAX_.ttf',
                font_size=60,
                color=color,
                stroke_color='black',
                stroke_width=10,
                margin=(200,200)
            ).with_position('center').with_start(subtitle['start']).with_end(subtitle['end'])
            
            text_clips.append(text_clip)
        
        # Combine everything
        final_clips = [video] + text_clips + [reddit_image]
        result = CompositeVideoClip(final_clips)
        
        # Long output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.long_output_dir / f"long_video_{timestamp}.mp4"
        
        result.write_videofile(output_file, fps=24)
        return output_file
    
    def stepY_crop_video(self, video_file):
        """Crop video file for purpose of making uploading Youtube shorts"""
        print(f"Step Y: Cropping Image")
        
        video = VideoFileClip(video_file)
        
        crop = Crop(width=video.h * (9/16), height=video.h, x_center=video.w/2, y_center=video.h/2)
        video = crop.apply(video)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.short_output_dir / f"short_video_{timestamp}.mp4"
        video.write_videofile(output_file)
        
        thumbnail_file = self.temp_dir / "thumbnail.png"
        video.save_frame(thumbnail_file, t=0)

        return output_file, thumbnail_file
    
    def step7_upload_youtube(self, video_file, title, description, tags):
        """Step 7: Upload to YouTube"""
        print("Step 7: Uploading to YouTube...")
        
        hashtags = " #shorts #youtubeshorts #viral #trending"
        extra_tags = ['shorts', 'viral', 'trending']
        
        try:
            # Load uploader and authenticate
            uploader = YoutubeUploader(secrets_file_path='credentials.json')
            uploader.authenticate(oauth_path='oauth.json')
            
            options = {
                "title":title.replace("|", " ") + hashtags,
                "description": description + hashtags,
                "tags": tags+extra_tags,
                "categoryId": "42",
                "privacyStatus": "public",
                "kids": False,
            }
            
            response = uploader.upload(video_file, options)
            
            uploader.close()
        
        except Exception as e:
            print(f"Error uploading video: {e}")
        
        return response
    
    def run_pipeline(self):
        """Run the complete pipeline"""
        print("Starting Simple Pipeline...")
        print("=" * 50)
        
        try:
            # Step 1: Generate story
            story, title, description, tags, gender = self.step1_generate_story()
            
            # Step X: Generate Reddit Image
            reddit_image = self.stepX_generate_reddit_image(title)
            
            # Step 2: Text to speech
            audio, audio_file, duration = self.step2_text_to_speech(story, title, gender)
            
            # Step 3: Add music
            audio_with_music = self.step3_add_music(audio_file, duration)
            
            # Step 4: Select video
            clip_file = self.step4_select_video(duration)
            if not clip_file:
                print("No video found - add videos to minecraft_videos/ directory")
                return None
            
            # Step 5: Create captions
            captions = self.step5_create_captions(audio_with_music)
            
            # Step 6: Merge everything
            merged_video = self.step6_merge_video(clip_file, audio_with_music, captions, reddit_image)
            
            # Step Y: Crop video
            cropped_video, thumbnail_file = self.stepY_crop_video(merged_video)
            
            # Step 7: Upload
            youtube_url = self.step7_upload_youtube(cropped_video, title, description, tags)
            
            # Clear temp directory
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            
            print("=" * 50)
            print("Pipeline completed!")
            print(f"Result: {youtube_url}")
            
            return youtube_url
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            return None

def main():
    """Test the pipeline"""
    pipeline = SimplePipeline()
    result = pipeline.run_pipeline()
    
    if result:
        print(f"\n Success! Video URL: {result}")
    else:
        print("\nPipeline failed. Check the steps above.")

if __name__ == "__main__":
    main()


# Test individual steps
def test_step(step_number):
    """Test individual pipeline steps"""
    pipeline = SimplePipeline()
    
    if step_number == 1:
        return pipeline.step1_generate_story()
    elif step_number == 2:
        story, title, description, tags, gender = pipeline.step1_generate_story()
        audio, audio_file = pipeline.step2_text_to_speech(story, title, gender)
        return audio
    elif step_number == 3:
        audio = AudioSegment.from_file(pipeline.output_dir / "story_audio_20250602_214117.mp3")
        duration = len(audio)
        return pipeline.step3_add_music("output/story_audio_20250602_214117.mp3", duration)
    elif step_number == 4:
        audio = AudioSegment.from_file(pipeline.output_dir / "story_audio_20250602_214117.mp3")
        duration = len(audio)
        return pipeline.step4_select_video(duration)
    elif step_number == 5:
        return pipeline.step5_create_captions("output/story_audio_20250602_214117.mp3")
    elif step_number == 6:
        audio_file = pipeline.output_dir / "story_audio_20250602_214117.mp3"
        clip_file = "clip.mp4"
        captions =  pipeline.step5_create_captions("output/story_audio_20250602_214117.mp3")
        return pipeline.step6_merge_video(clip_file, audio_file, captions)
    elif step_number == 7:
        return pipeline.step7_upload_youtube("output.mp4", "Test", "Test", ["test"])
    elif step_number == 8:
        return pipeline.stepX_generate_reddit_image("Curtain Toga|Confessions")
    elif step_number == 9:
        return pipeline.stepY_crop_video('output.mp4')
    else:
        print("Invalid step number (1-7)")


