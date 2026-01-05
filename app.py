# app.py
import streamlit as st
import numpy as np
import librosa
import json
import tensorflow as tf
from data_preprocess import load_audio, compute_log_mel, SAMPLE_RATE
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import av
import queue
import os

MODEL_PATH = "models/emotion_model.h5"
LABEL_MAP = "models/label_map.json"

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP, 'r') as f:
        label_data = json.load(f)
    classes = label_data['classes']
    return model, classes

model, classes = load_model()

st.title("🎙️ Speech Emotion Recognition")
st.write("Record or upload audio (<= 3 seconds recommended). Model predicts emotion.")

col1, col2 = st.columns([1,1])

# Upload option
with col1:
    uploaded_file = st.file_uploader("Upload a WAV file", type=["wav","mp3","ogg","flac"])
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        # write to temp file
        tmp_path = "tmp_input.wav"
        with open(tmp_path, "wb") as f:
            f.write(bytes_data)
        y = load_audio(tmp_path)
        st.audio(bytes_data, format='audio/wav')
        os.remove(tmp_path)
        # preproc & predict
        log_mel = compute_log_mel(y)
        X = np.expand_dims(log_mel, axis=(0,-1))  # (1, n_mels, t, 1)
        preds = model.predict(X)[0]
        top_idx = np.argmax(preds)
        st.subheader(f"Prediction: {classes[top_idx]} ({preds[top_idx]*100:.2f}%)")
        # show probabilities
        prob_df = {classes[i]: float(preds[i]) for i in range(len(classes))}
        st.json(prob_df)

# Recording option via webrtc
with col2:
    st.markdown("**Or record live (click Start)**")

audio_q = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.buffer = np.zeros((0,), dtype='float32')

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        # receive audio frames from browser
        pcm = frame.to_ndarray()  # shape (channels, samples)
        # convert to mono
        if pcm.ndim > 1:
            pcm = np.mean(pcm, axis=0)
        # normalize from int16 to float32 if necessary
        pcm_f = pcm.astype(np.float32) / np.max(np.abs(pcm)) if np.max(np.abs(pcm)) != 0 else pcm.astype(np.float32)
        self.buffer = np.concatenate((self.buffer, pcm_f))
        # we don't send frames back; just return the input
        return frame

webrtc_ctx = webrtc_streamer(key="speech-emotion", mode=WebRtcMode.SENDRECV,
                             audio_processor_factory=AudioProcessor,
                             media_stream_constraints={"audio": True, "video": False})

if webrtc_ctx.state.playing:
    st.info("Recording... Click Stop when done.")
if st.button("Process Recorded Audio"):
    # access audio buffer via webrtc_ctx? The streamlit-webrtc example recommended to implement a shared object
    # A simple approach: write small snippet that saves last N seconds from the AudioProcessor instance.
    try:
        ap = webrtc_ctx.audio_processor
        buffer = ap.buffer  # numpy array
        if buffer is None or len(buffer) == 0:
            st.error("No audio recorded yet. Press Start and speak.")
        else:
            # ensure length at least SAMPLE_RATE*DURATION
            if len(buffer) < SAMPLE_RATE * 1:  # if very short, warn
                st.warning("Very short audio recorded; results may be noisy.")
            # trim/pad
            if len(buffer) < SAMPLE_RATE * 3:
                pad_len = int(SAMPLE_RATE*3) - len(buffer)
                buffer = np.pad(buffer, (0,pad_len))
            else:
                buffer = buffer[:int(SAMPLE_RATE*3)]
            # predict
            log_mel = compute_log_mel(buffer)
            X = np.expand_dims(log_mel, axis=(0,-1))
            preds = model.predict(X)[0]
            top_idx = np.argmax(preds)
            st.subheader(f"Prediction: {classes[top_idx]} ({preds[top_idx]*100:.2f}%)")
            st.json({classes[i]: float(preds[i]) for i in range(len(classes))})
    except Exception as e:
        st.error(f"Error processing audio: {e}")

# Optional: show an example spectrogram
st.markdown("---")
st.write("Notes: Use a quiet environment, speak clearly. Longer audio may be trimmed/padded to fixed length.")