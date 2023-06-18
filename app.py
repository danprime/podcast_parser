import streamlit as st
import whisper
import os
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

    st.info('Breaking podcast into 5 minute chunks.')
    #break into 5 minute chunks
    chunk_length_five_minutes = 5 * 60 * 1000
    podcast_chunks = podcast[::chunk_length_five_minutes]

    st.info('Transcribe')

    #transcriptions = []
    
    #for i, chunk in enumerate(podcast_chunks):
    #    chunk.export(f'output/chunk_{i}.mp4', format='mp4')
    
    # following blogpost here: https://huggingface.co/blog/asr-chunking
    transcribe_pipe = pipeline(model="facebook/wav2vec2-base-960h")
    transcription = transcribe_pipe(audiofile, chunk_length_s=10, stride_length_s=(4, 2))

    st.session_state['transcription'] = transcription
    print(f"transcription: {transcription}")
    st.info('Done Transcription')

    return transcription

st.markdown("# Podcast Q&amp;A")

st.markdown(
        """
        This helps understand information-dense podcast episodes by doing the following:
        - Speech to Text transcription - using OpenSource Whisper Model
        - Summarizes the episode
        - Allows you to ask questions and returns direct quotes from the episode.

        """
        )

if st.button("Process Audio File"):
    transcribe_audio("marketplace-2023-06-14.mp3")

#audio_file = st.file_uploader("Upload audio copy of file", key="upload", type=['.mp3'])


# if audio_file:
#    transcribe_audio(audio_file)
