#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# In[8]:


# Path to dataset
data_dir = 'C:\\Users\\ashud\\Downloads\\genres\\genres'


# In[9]:


# Function to extract mel spectrogram features from an audio file
def extract_mel_spectrogram(file_path, target_length=1300, augment=False):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if augment:
        # Random pitch shifting
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
        # Random time stretching
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    


# In[11]:


# Function to extract mel spectrogram features from an audio file
def extract_mel_spectrogram(file_path, target_length=1300, augment=False):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    if augment:
        # Random pitch shifting
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
        # Random time stretching
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to dB scale
    
    # Padding or truncating the spectrogram to the target length
    if mel_spec_db.shape[1] < target_length:
        # Padding if the length is shorter
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_length - mel_spec_db.shape[1])), mode='constant')
    elif mel_spec_db.shape[1] > target_length:
        # Truncating if the length is longer
        mel_spec_db = mel_spec_db[:, :target_length]
    
    return mel_spec_db

# Prepare data
features = []
labels = []

target_length = 1300  # Set the target length for consistent spectrogram size

for genre in os.listdir(data_dir):
    genre_path = os.path.join(data_dir, genre)
    if os.path.isdir(genre_path):
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            if os.path.isfile(file_path):
                mel_spectrogram = extract_mel_spectrogram(file_path, target_length=target_length, augment=True)
                features.append(mel_spectrogram)
                labels.append(genre)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Normalize features to range [0, 1]
features = (features - features.min()) / (features.max() - features.min())

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)  # One-hot encode labels

# Reshape features to include channel dimension for CNN input
features = features[..., np.newaxis]


# In[12]:


# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)


# In[14]:


# Build a 2D CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(128, target_length, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(labels.shape[1], activation='softmax')
])


# In[15]:


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[17]:


# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.2)


# In[18]:


# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# In[19]:


# Save the model
model.save('music_genre_cnn_melspectrogram.h5')


# In[ ]:




