import streamlit as st
from transformers import pipeline

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

