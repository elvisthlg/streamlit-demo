import streamlit as st
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import os
import tempfile
from io import BytesIO

st.set_page_config(page_title="Audio Testing App", layout="wide")

st.title("Audio Testing App")

# Sidebar for API Keys
st.sidebar.header("API Keys")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

if elevenlabs_api_key:
    os.environ["ELEVENLABS_API_KEY"] = elevenlabs_api_key

# Tabs for different models
tab1, tab2 = st.tabs(["Transcription (OpenAI)", "Text-to-Speech (ElevenLabs)"])

with tab1:
    st.header("OpenAI Transcription")
    
    transcription_model = st.selectbox(
        "Select Transcription Model",
        ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"]
    )
    
    uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'wav', 'm4a', 'ogg', 'flac'])
    
    if st.button("Transcribe"):
        if not openai_api_key:
            st.error("Please provide an OpenAI API Key in the sidebar.")
        elif not uploaded_file:
            st.error("Please upload an audio file.")
        else:
            with st.spinner("Transcribing..."):
                try:
                    client = OpenAI(api_key=openai_api_key)
                    # We need to save the uploaded file to a temporary file because OpenAI client expects a file-like object with a name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    with open(tmp_file_path, "rb") as audio_file:
                        transcription = client.audio.transcriptions.create(
                            model=transcription_model,
                            file=audio_file,
                            response_format="text"
                        )
                        
                    st.success("Transcription Complete!")
                    st.text_area("Transcription Result", transcription, height=300)
                    
                    # Cleanup
                    os.remove(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")

with tab2:
    st.header("ElevenLabs Text-to-Speech")
    
    tts_text = st.text_area("Text to convert to speech", "Hello! This is a test of the ElevenLabs Text-to-Speech system.", height=150)
    
    # Static list of some popular ElevenLabs voices. 
    # Can also fetch dynamically using API if needed, but static is faster.
    elevenlabs_voices = ["Rachel", "Drew", "Clyde", "Paul", "Domi", "Fin", "Bella", "Antoni", "Thomas", "Charlie", "Emily", "Elli", "Callum", "Patrick", "Harry", "Liam", "Dorothy", "Josh", "Arnold", "Charlotte", "Matilda", "Matthew", "James", "Joseph", "Jeremy", "Michael", "Ethan", "Gigi", "Freya", "Grace", "Daniel", "Serena", "Mimi"]
    
    voice_selection = st.selectbox("Select Voice", elevenlabs_voices)
    
    # Model option
    elevenlabs_model = st.selectbox("Select TTS Model", ["eleven_multilingual_v2", "eleven_monolingual_v1", "eleven_english_v2"])
    
    if st.button("Generate Speech"):
        if not elevenlabs_api_key:
            st.error("Please provide an ElevenLabs API Key in the sidebar.")
        elif not tts_text.strip():
            st.error("Please enter some text.")
        else:
            with st.spinner("Generating speech..."):
                try:
                    client = ElevenLabs(api_key=elevenlabs_api_key)
                    
                    audio_generator = client.generate(
                        text=tts_text,
                        voice=voice_selection,
                        model=elevenlabs_model
                    )
                    
                    # Consume the generator into bytes
                    audio_bytes = b"".join(chunk for chunk in audio_generator if chunk)
                    
                    st.success("Speech Generated!")
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    # Add download button
                    st.download_button(
                        label="Download Audio",
                        data=audio_bytes,
                        file_name="generated_speech.mp3",
                        mime="audio/mp3"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
