---
title: Podcast Parser
emoji: 🔥
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.21.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Podcast Parser
This uses Whisper AI and Flan T5 to transcribe and summarize a podcast (MP3).

## How to use
1. Setup a new space on HuggingFace 
2. Import code
3. Run
4. Click "Parse Audio File" button

## What's happening in the background
1. Takes the mp3 file and uses the Whisper Model (small.en) to transcribe it with a window (because it's a very long file)
2. Takes the transcription of the audio file and then summarizes it using Flan T5
3. Joins all the summary into text
4. Presents everything using Streamlit

## Note
This is a proof of concept. It has not been optimized at all.

This is open source code under Apache 2.0 License.
