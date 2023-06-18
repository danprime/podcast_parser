import streamlit as st
import whisper
import os
import torch
from transformers import pipeline
from pydub import AudioSegment

def transcribe_audio(audiofile):

    st.session_state['audio'] = audiofile
    print(f"audio_file_session_state:{st.session_state['audio'] }")

    #get size of audio file
    audio_size = round(os.path.getsize(st.session_state['audio'])/(1024*1024),1)
    print(f"audio file size:{audio_size}")

    #determine audio duration
    podcast = AudioSegment.from_mp3(st.session_state['audio'])
    st.session_state['audio_segment'] = podcast
    podcast_duration = podcast.duration_seconds
    print(f"Audio Duration: {podcast_duration}")

    st.info('Transcribing...')
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-medium",
    chunk_length_s=30,
    device=device,
    max_new_tokens=60,
    )

    transcription = pipe(audiofile, batch_size=8)["text"]

    st.session_state['transcription'] = transcription
    print(f"transcription: {transcription}")
    st.info('Done Transcription')

    return transcription

def summarize_podcast(audiotranscription):
    sum_pipe = pipeline("summarization",model="google/flan-t5-base",clean_up_tokenization_spaces=True)
    summarized_text = sum_pipe(audiotranscription,
                               max_length=1000,
                               min_length=100,
                               do_sample=False, 
                               early_stopping=True,
                              num_beams=4)
    summarized_text = ' '.join([summ['summary_text'] for summ in summarized_text])

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
    transcribe_audio("marketplace-2023-06-14.mp3")

#audio_file = st.file_uploader("Upload audio copy of file", key="upload", type=['.mp3'])


# if audio_file:
#    transcribe_audio(audio_file)
