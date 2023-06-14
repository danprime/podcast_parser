import streamlit as st
import transformers import pipeline

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)

pipe = pipeline('sentiment-analysis')
text = st.text_area('enter some text')

if text:
    out = pipe(text)
    st.json(out)
