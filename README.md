# Speech-Emotion-Recognition-System

## Overview: 
This project implements an end-to-end Speech Emotion Recognition (SER) system using deep learning to classify human emotions from speech audio. The system processes raw audio input, extracts meaningful features, predicts emotions using a trained neural network, and provides real-time inference through a Streamlit web application.
The project demonstrates the complete machine learning lifecycle — data preprocessing, model training, inference, and deployment-ready UI.

## Objective:
- Automatically recognize emotions from speech audio.
- Learn emotional patterns using deep learning.
- Provide a real-time, user-friendly web interface.

## Emotions Classified:
Angry
Calm
Disgust
Fearful
Happy
Neutral
Sad
Surprised

## Dataset:
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
High-quality, labeled emotional speech recordings.
Kaggle link: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

## Methodology:
1. Audio Preprocessing
- Converted audio to mono
- Standardized sampling rate and duration
- Removed inconsistencies across samples

2. Feature Extraction
- Extracted Log-Mel Spectrograms using Librosa
- Converted raw audio signals into 2D time–frequency representations

3. Model Architecture
- CNN layers to learn spectral (frequency-based) features
- LSTM layers to capture temporal dependencies in speech
- Softmax output layer for multi-class emotion classification

4. Model Training
- Loss function: Categorical Cross-Entropy
- Optimizer: Adam
- Validation-based early stopping

5. Inference & UI
- Emotion prediction with confidence scores
- Interactive visualization using Streamlit

## How to Run the Project
1. Create Virtual Environment (Python 3.10)
py -3.10 -m venv ser-venv
.\ser-venv\Scripts\activate

2. Install Dependencies
pip install -r requirements.txt

3. Preprocess Data
python data_preprocess.py

4. Train Model
python train_model.py

5. Run Streamlit App
streamlit run app.py

## Technologies Used
- Python 3.10
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Streamlit
- Scikit-learn

## Applications
- Call center emotion analytics
- Mental health monitoring tools
- Human–computer interaction systems
- Smart assistants and voice AI
- Emotion-aware gaming and entertainment
