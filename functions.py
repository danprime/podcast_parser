import whisper
import os

@st.cache_data
def transcribe_audio(audiofile):

    st.session_state['audio'] = audiofile

    print(f"audio_file_session_state:{st.session_state['audio'] }")

    #get size of audio file
    audio_size = round(os.path.getsize(st.session_state['audio'])/(1024*1024),1)

    print(f"audio file size:{audio_size}")


    return audio_size
