#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Path to your dataset
data_dir = 'C:\\Users\\ashud\\Downloads\\genres\\genres'

# Function to augment audio (pitch shifting and time stretching)
def augment_audio(y, sr):
    # Random pitch shifting
    if np.random.rand() < 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
    # Random time stretching
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    return y

# Function to extract features from an audio file
def extract_features(file_path, augment=False):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if augment:
        y = augment_audio(y, sr)  # Apply augmentation
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# Prepare training data
features = []
labels = []

# Loop through the dataset and extract features and labels
for genre in os.listdir(data_dir):
    genre_path = os.path.join(data_dir, genre)
    if os.path.isdir(genre_path):
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            if os.path.isfile(file_path):
                features.append(extract_features(file_path, augment=True))  # Augmentation enabled
                labels.append(genre)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model
joblib.dump(model, 'audio_genre_classifier_rf_with_augmentation.pkl')


# In[ ]:




