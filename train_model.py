# train_model.py
import os
import numpy as np
from keras import layers, models, callbacks, utils
from data_preprocess import load_audio, compute_log_mel
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 1. Build dataset (example: walk folder and parse emotion from filename)
def build_dataset(root_dir):
    X, y = [], []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if not fname.endswith('.wav'): continue
            path = os.path.join(root, fname)
            # RAVDESS filenames contain emotion label in naming convention:
            # e.g., '03-01-05-01-01-01-01.wav' – you will parse based on dataset readme
            # For simplicity let's assume you have mapping function to parse emotion label
            label = parse_ravdess_emotion(fname)
            y.append(label)
            y_audio = load_audio(path)
            log_mel = compute_log_mel(y_audio)
            # ensure a fixed-size (n_mels, time) -> you can resize/pad/truncate
            X.append(log_mel)
    X = np.array(X)  # shape (N, n_mels, time)
    return X, y

# parse function (implement according to dataset naming)
def parse_ravdess_emotion(fname):
    # mapping from RAVDESS emotion code
    code = int(fname.split('-')[2])
    mapping = {1:'neutral',2:'calm',3:'happy',4:'sad',5:'angry',6:'fearful',7:'disgust',8:'surprised'}
    return mapping.get(code, 'neutral')

# Build dataset and labels
X, y = build_dataset('data/ravdess')
X = X[..., np.newaxis]  # add channel
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = utils.to_categorical(y_enc)

# Split (speaker-wise split is recommended; this simple split is illustrative)
X_train, X_val, y_train, y_val = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# Build model
def build_cnn_lstm(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D((2,2))(x)
    # collapse freq dimension for LSTM across time dimension
    shape = x.shape
    x = layers.Permute((2,1,3))(x)  # time major
    x = layers.Reshape((x.shape[1], x.shape[2]*x.shape[3]))(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)
    return model

model = build_cnn_lstm(X_train.shape[1:], y_train.shape[1])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
cb = [
    callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
    callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
]

history = model.fit(X_train, y_train, validation_data=(X_val,y_val), epochs=60, batch_size=32, callbacks=cb)
model.save('models/emotion_model.h5')

# Save label encoder classes
with open('models/label_map.json', 'w') as f:
    json.dump({'classes': le.classes_.tolist()}, f)