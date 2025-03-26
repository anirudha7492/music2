# Music Genre Identification

This project implements a **Music Genre Identification** system using **Convolutional Neural Networks (CNNs)**. The goal is to classify audio files into their respective genres by analyzing their Mel spectrogram representations.

## Dataset
- The dataset consists of audio files from different music genres.
- Download the dataset from the given link: [Download Here](https://www.dropbox.com/s/4jw31k5mlzcmgis/genres.tar.gz?dl=0)
- Extract the contents to your preferred directory.

## Project Structure
```bash
- genres/            # Dataset containing audio files organized by genre
- music_genre_classification.py   # Main Python script
- README.md           # Project documentation
- accuracy_plot.png   # Training and validation accuracy plot
- loss_plot.png       # Training and validation loss plot
- evaluation_report.txt # Classification report with performance metrics
```

## Requirements
Install the following dependencies using `pip`:
```bash
pip install numpy librosa matplotlib scikit-learn tensorflow
```

## Code Explanation

### 1. Import Libraries
The first section imports essential libraries like **Librosa** for audio processing, **Matplotlib** for plotting, **TensorFlow** for building neural networks, and **Scikit-Learn** for data preprocessing and evaluation.

### 2. Extract Mel Spectrograms
```python
mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
```
- **Librosa** extracts Mel spectrograms, representing audio signals in the frequency domain.
- These spectrograms are converted to decibel (dB) scale for better visual and analytical interpretation.

### 3. Data Augmentation
```python
if augment:
    if np.random.rand() < 0.5:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2))
    if np.random.rand() < 0.5:
        y = librosa.effects.time_stretch(y, rate=np.random.uniform(0.8, 1.2))
```
- Data augmentation techniques such as **pitch shifting** and **time stretching** are applied randomly to introduce variability and enhance model robustness.

### 4. Model Building
```python
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
```
- A **CNN** is designed using **Conv2D** layers to extract spatial features from spectrogram images.
- **MaxPooling2D** reduces the dimensionality, and **Dropout** prevents overfitting.
- The final output layer uses a **softmax** activation for multiclass classification.

### 5. Evaluation and Performance Visualization
```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.savefig('accuracy_plot.png')
```
- After training, accuracy and loss curves are plotted to visualize the model’s learning process.

## Classification Report

Below is the classification report generated from the test data:

```
              precision    recall  f1-score   support

       blues       0.40      0.10      0.16        20
   classical       0.68      1.00      0.81        13
     country       0.33      0.04      0.07        27
       disco       0.36      0.38      0.37        21
      hiphop       0.44      0.27      0.33        15
        jazz       0.62      0.73      0.67        22
       metal       0.58      0.72      0.64        25
         pop       0.24      0.46      0.32        13
      reggae       0.40      0.43      0.42        23
        rock       0.20      0.33      0.25        21

    accuracy                           0.42       200
   macro avg       0.43      0.45      0.40       200
weighted avg       0.42      0.42      0.39       200
```

- **Precision**: Accuracy of positive predictions.
- **Recall**: Ability of the model to detect actual positives.
- **F1-score**: Harmonic mean of precision and recall.
- The overall accuracy achieved was **42%**.

