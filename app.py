import streamlit as st
import whisper
import os
import torch
from transformers import pipeline
from pydub import AudioSegment

def transcribe_audio(audiofile):

    st.session_state['audio'] = audiofile
    print(f"audio_file_session_state:{st.session_state['audio'] }")

    st.info("Getting size of file")
    #get size of audio file
    audio_size = round(os.path.getsize(st.session_state['audio'])/(1024*1024),1)
    print(f"audio file size:{audio_size}")

    #determine audio duration
    podcast = AudioSegment.from_mp3(st.session_state['audio'])
    st.session_state['audio_segment'] = podcast
    podcast_duration = podcast.duration_seconds
    print(f"Audio Duration: {podcast_duration}")

    st.info("Transcribing")
    whisper_model = whisper.load_model("small.en")
    transcription = whisper_model.transcribe(audiofile)
    st.session_state['transcription'] = transcription
    print(f"ranscription: {transcription['text']}")
    st.info('Done Transcription')

    return transcription

def summarize_podcast(audiotranscription):
    summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum", device=0)

    summarized_text = summarizer(audiotranscription)
    return summarized_text
    

st.markdown("# Podcast Q&amp;A")

st.markdown(
        """
        This helps understand information-dense podcast episodes by doing the following:
        - Speech to Text transcription - using OpenSource Whisper Model
        - Summarizes the episode
        - Allows you to ask questions and returns direct quotes from the episode.

        """
        )

st.audio("marketplace-2023-06-14.mp3") 
if st.button("Process Audio File"):
    podcast_text = transcribe_audio("marketplace-2023-06-14.mp3")
    #write text out
    with st.expander("See Transcription"):
        st.caption(podcast_text)
    
    #Summarize Text
    podcast_summary = summarize_podcast(podcast_text)
    st.markdown(
        """
           ##Summary of Text
        """
        )
    st.text(podcast_summary)

if st.button("Summarize Podcast"):
    with open('transcription.txt', 'r') as file:
        podcast_summary = file.read().rstrip()
    podcast_summary = summarize_podcast(podcast_text)
    st.markdown(
        """
           ##Summary of Text
        """
        )
    st.text(podcast_summary)

#audio_file = st.file_uploader("Upload audio copy of file", key="upload", type=['.mp3'])


# if audio_file:
#    transcribe_audio(audio_file)
