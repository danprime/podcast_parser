import streamlit as st
from transformers import pipeline

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

pipe = pipeline("summarization", model='facebook/bart-large-cnn')
text = st.text_area('enter some text')

if text:
    out = pipe(text, max_length=130, min_length=30, do_sample=False)
    st.json(out)
