import streamlit as st
from transformers import pipeline

def transcribe_audio(audiofile):

    st.session_state['audio'] = audiofile

    print(f"audio_file_session_state:{st.session_state['audio'] }")

    #get size of audio file
    audio_size = round(os.path.getsize(st.session_state['audio'])/(1024*1024),1)

    print(f"audio file size:{audio_size}")

    return audio_size

st.markdown("# Podcast Q&amp;A")

st.markdown(
        """
        This helps understand information-dense podcast episodes by doing the following:
        - Speech to Text transcription - using OpenSource Whisper Model
        - Summarizes the episode
        - Allows you to ask questions and returns direct quotes from the episode.

        """
        )

audio_file = st.file_uploader("Upload audio copy of file", key="upload", type=['.mp3'])


if audio_file:
    transcribe_audio(audio_file)
