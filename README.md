# Story-to-Video Pipeline

Automated pipeline that generates AI stories and turns them into videos with gameplay footage, voice narration, and captions.

## Features

- **AI Story Generation** with OpenAI GPT
- **Text-to-Speech** using ElevenLabs
- **Auto Video Editing** with MoviePy
- **Caption Generation** via Whisper
- **YouTube Upload** automation

## Quick Start

1. **Install dependencies**
   ```bash
   pip install openai elevenlabs moviepy whisper-openai
   ```

2. **Add videos to `minecraft_videos/` folder**

3. **Run once to create config, then add your API keys**
   ```bash
   python shortFormed.py
   # Edit config.json with your API keys
   ```

4. **Run the pipeline**
   ```bash
   python shortFormed.py
   ```

## Configuration

Edit `config.json`:
```json
{
  "openai_api_key": "your_key_here",
  "elevenlabs_api_key": "your_key_here",
  "story_theme": "absurd office situations",
}
```

## Pipeline Steps

1. Generate story with AI
2. Convert text to speech
3. Add background music (optional)
4. Select video from library
5. Create synchronized captions
6. Merge video, audio, and captions
7. Upload to YouTube

## Development Status

This is a skeleton implementation. Core structure is complete, but individual steps need implementation:

- [ ] OpenAI story generation
- [ ] ElevenLabs TTS
- [ ] MoviePy video editing
- [ ] Whisper transcription
- [ ] YouTube API upload

## Test Individual Steps

```python
from shortFormed import test_step
story = test_step(1)  # Test story generation
video = test_step(4)  # Test video selection
```

## Requirements

- Python 3.7+
- OpenAI API key
- ElevenLabs API key
- YouTube API credentials
