# data_preprocess.py
import librosa
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3.0         # seconds - fixed window
SAMPLES = int(SAMPLE_RATE * DURATION)
N_MELS = 64

def load_audio(path, sr=SAMPLE_RATE, duration=DURATION, mono=True):
    y, orig_sr = librosa.load(path, sr=None, mono=mono)
    # resample if needed
    if orig_sr != sr:
        y = librosa.resample(y=y, orig_sr=orig_sr, target_sr=sr)
    # pad/truncate
    if len(y) < int(sr * duration):
        pad_len = int(sr * duration) - len(y)
        y = np.pad(y, (0, pad_len))
    else:
        y = y[:int(sr * duration)]
    return y

def compute_log_mel(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=256):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    log_mel = librosa.power_to_db(mel)
    return log_mel  # shape (n_mels, time_frames)

def compute_mfcc(y, sr=SAMPLE_RATE, n_mfcc=40):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc  # shape (n_mfcc, time_frames)streamlit run app.py
