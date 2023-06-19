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
def chunk_and_preprocess_text(text, model_name= 'philschmid/flan-t5-base-samsum'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sentences = sent_tokenize(text)

    length = 0
    chunk = ""
    chunks = []
    count = -1
    
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter
    
        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter
    
            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk) # save the chunk
      
        else: 
            chunks.append(chunk) # save the chunk
            # reset 
            length = 0 
            chunk = ""
        
            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))

    return chunks

def summarize_podcast(audiotranscription):
    st.info("Summarizing...")
    summarizer = pipeline("summarization", model="philschmid/flan-t5-base-samsum", device=0)

    st.info("Chunking text")
    text_chunks = chunk_and_preprocess_text(audiotranscription)

    summarized_text = summarizer(text_chunks, max_len=200,min_len=50)
    st.session_state['summary'] = summarized_text
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
        st.caption(podcast_text['text']})
    
    #Summarize Text
    podcast_summary = summarize_podcast(podcast_text['text'])
    st.markdown(
        """
           ##Summary of Text
        """
        )
    st.text(podcast_summary['summary_text'])

if st.button("Summarize Podcast"):
    with open('transcription.txt', 'r') as file:
        podcast_text = file.read().rstrip()
    podcast_summary = summarize_podcast(podcast_text)
    st.markdown(
        """
           ##Summary of Text
        """
        )
    st.text(podcast_summary['summary_text'])

#audio_file = st.file_uploader("Upload audio copy of file", key="upload", type=['.mp3'])


# if audio_file:
#    transcribe_audio(audio_file)
