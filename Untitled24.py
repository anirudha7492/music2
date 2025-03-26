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
from sklearn.metrics import classification_report

# Path to dataset
data_dir = 'C:\\Users\\ashud\\Downloads\\genres\\genres'

def extract_mel_spectrogram(file_path, target_length=1300, augment=False):
    """
    Extracts the Mel spectrogram from an audio file using Librosa.
    
    Parameters:
        file_path (str): Path to the audio file.
        target_length (int): Target spectrogram length for consistency.
        augment (bool): Apply data augmentation like pitch shifting and time stretching.
    
    Returns:
        np.ndarray: Mel spectrogram in dB scale.
    """
    y, sr = librosa.load(file_path, sr=None, mono=True)
    
    if augment:
        if np.random.rand() < 0.5:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
        if np.random.rand() < 0.5:
            y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))

    # Generate Mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Padding or truncating to ensure uniform input size
    if mel_spec_db.shape[1] < target_length:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, target_length - mel_spec_db.shape[1])), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :target_length]

    return mel_spec_db

# Data Preparation
features, labels = [], []
target_length = 1300

for genre in os.listdir(data_dir):
    genre_path = os.path.join(data_dir, genre)
    if os.path.isdir(genre_path):
        for file_name in os.listdir(genre_path):
            file_path = os.path.join(genre_path, file_name)
            if os.path.isfile(file_path):
                mel_spectrogram = extract_mel_spectrogram(file_path, target_length=target_length, augment=True)
                features.append(mel_spectrogram)
                labels.append(genre)

features = np.array(features)
labels = np.array(labels)

# Normalization to scale between 0 and 1
features = (features - features.min()) / (features.max() - features.min())

# Encode labels
encoder = LabelEncoder()
labels = encoder.fit_transform(labels)
labels = to_categorical(labels)

# Reshape for CNN input
features = features[..., np.newaxis]

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Model Building
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

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model Training
history = model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.2)

# Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Save the Model
model.save('music_genre_cnn_melspectrogram.h5')

# Classification Report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred_classes, target_names=encoder.classes_))

# Plot Performance Figures
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()





