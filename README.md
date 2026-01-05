# Speech-Emotion-Recognition-System
Overview: 
This project implements an end-to-end Speech Emotion Recognition (SER) system using deep learning to classify human emotions from speech audio. The system processes raw audio input, extracts meaningful features, predicts emotions using a trained neural network, and provides real-time inference through a Streamlit web application.
The project demonstrates the complete machine learning lifecycle — data preprocessing, model training, inference, and deployment-ready UI.

Objective:
* Automatically recognize emotions from speech audio.
* Learn emotional patterns using deep learning.
* Provide a real-time, user-friendly web interface.

Emotions Classified:
Angry
Calm
Disgust
Fearful
Happy
Neutral
Sad
Surprised

Dataset:
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
High-quality, labeled emotional speech recordings.
Kaggle link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Methodology:
1. Audio Preprocessing
* Converted audio to mono
* Standardized sampling rate and duration
* Removed inconsistencies across samples

2. Feature Extraction
* Extracted Log-Mel Spectrograms using Librosa
* Converted raw audio signals into 2D time–frequency representations
