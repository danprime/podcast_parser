import streamlit as st
import whisper
import os
import torch
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from pydub import AudioSegment
from nltk import sent_tokenize
nltk.download('punkt')

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

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
    print(f"transcription: {transcription['text']}")
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

    #summarized_text = summarizer(text_chunks, max_len=200,min_len=50)
    summarized_text = summarizer(text_chunks)
    st.session_state['summary'] = summarized_text
    print(f"Summary: {summarized_text}")
    #summarized_text is an array of objects with key summary_text
    full_summary = ' '.join(item['summary_text'] for item in summarized_text)
    return full_summary

def prepare_text_for_qa(audiotranscription):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20)
    documents = text_splitter.split_documents(audiotranscription)
    revalue = ""
    return revalue

st.markdown("# Podcast Summarizer")

st.markdown(
        """
        This helps understand information-dense podcast episodes by doing the following:
        - Speech to Text transcription - using OpenSource Whisper Model (small.en)
        - Summarizes the episode - using philschmid/flan-t5-base-samsum a model based on Google's flan t5

        - As a proof of Concept: the Podcast Episode of Marketplace Business News Podcast from NPR on June 14 is used in this codebase.
        - The file is THE ONLY HARDCODED piece of information used in this application.

        - *HOW TO TEST:* Click on "Process Audio File" button

        """
        )

st.text("Marketplace Episode June 14 2023")
st.audio("marketplace-2023-06-14.mp3") 
if st.button("Process Audio File"):
    podcast_text = transcribe_audio("marketplace-2023-06-14.mp3")
    #write text out
    with st.expander("See Transcription"):
        st.caption(podcast_text['text'])
    
    #Summarize Text
    podcast_summary = summarize_podcast(podcast_text['text'])
    st.markdown(
        """
           ## Summary of Text
        """
        )
    st.text(podcast_summary)

# if st.button("Summarize Podcast"):
    # with open('transcription.txt', 'r') as file:
        # podcast_text = file.read().rstrip()
    # podcast_summary = summarize_podcast(podcast_text)
    # st.markdown(
        # """
           # ## Summary of Text
        # """
        # )
    # st.text(podcast_summary)
