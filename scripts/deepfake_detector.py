import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from google.colab import drive
import zipfile
import shutil


# Set paths for Celeb-DF dataset (adjust these paths according to your Google Drive structure)
base_path = '/content/drive/MyDrive'
dataset_zip = os.path.join(base_path, 'Celeb-DF-v2.zip')

# Create a temporary directory to extract files
temp_dir = '/content/temp_celeb_df'
os.makedirs(temp_dir, exist_ok=True)

# Extract the zip file
print("Extracting dataset...")
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(temp_dir)

# Update paths to use the extracted files
real_videos_path = os.path.join(temp_dir, 'Celeb-real')
fake_videos_path = os.path.join(temp_dir, 'Celeb-synthesis')

# Video listing
real_videos = [os.path.join(real_videos_path, video) for video in os.listdir(real_videos_path) if video.endswith('.mp4')]
fake_videos = [os.path.join(fake_videos_path, video) for video in os.listdir(fake_videos_path) if video.endswith('.mp4')]

video_paths = real_videos + fake_videos
labels = [0] * len(real_videos) + [1] * len(fake_videos)

df = pd.DataFrame({'video_path': video_paths, 'label': labels})

# Data splitting
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

def build_lstm_model(num_frames=10):
    input_layer = layers.Input(shape=(num_frames, 112, 112, 3))

    # CNN layers
    x = layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu'))(input_layer)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), activation='relu'))(x)
    x = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(x)
    x = layers.TimeDistributed(layers.Flatten())(x)

    # LSTM layers
    x = layers.LSTM(256, return_sequences=True)(x)
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    x = layers.Dropout(0.5)(x)

    # Dense layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def data_generator(df, batch_size, num_frames=10):
    num_samples = len(df)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = df.iloc[offset:offset + batch_size]
            video_sequences = []
            labels = []

            for _, row in batch_samples.iterrows():
                video_path = row['video_path']
                label = row['label']
                frames = []

                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                for i in np.linspace(0, frame_count - 1, num_frames, dtype=int):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.resize(frame, (112, 112))
                        frame = frame / 255.0  # Normalize pixel values
                        frames.append(frame)
                    else:
                        frames.append(np.zeros((112, 112, 3)))

                cap.release()
                video_sequences.append(np.array(frames))
                labels.append(label)

            X = np.array(video_sequences)
            y = np.array(labels)
            yield X, y

# Build model
print("Building model...")
lstm_model = build_lstm_model(num_frames=10)
lstm_model.summary()

# Training parameters
batch_size = 8  # Adjust based on your GPU memory
num_frames = 10
epochs = 20

train_gen = data_generator(train_df, batch_size=batch_size, num_frames=num_frames)
val_gen = data_generator(val_df, batch_size=batch_size, num_frames=num_frames)

# Train model
print("Training model...")
try:
    history = lstm_model.fit(
        train_gen,
        steps_per_epoch=len(train_df) // batch_size,
        validation_data=val_gen,
        validation_steps=len(val_df) // batch_size,
        epochs=epochs,
        verbose=1
    )

    # Save model
    model_save_path = '/content/drive/MyDrive/lstm_model.h5'
    lstm_model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

except Exception as e:
    print(f"Error during model training: {e}")

# Evaluate the model
print("Evaluating model...")
val_loss, val_accuracy = lstm_model.evaluate(
    val_gen,
    steps=len(val_df) // batch_size,
    verbose=1
)

print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Generate classification report
print("Generating classification report...")
y_true = []
y_pred = []

test_gen = data_generator(test_df, batch_size=batch_size, num_frames=num_frames)
for _ in range(len(test_df) // batch_size):
    X_batch, y_batch = next(test_gen)
    preds = lstm_model.predict(X_batch)
    y_true.extend(y_batch)
    y_pred.extend((preds > 0.5).astype(int))

print(classification_report(y_true, y_pred))

# Function to predict on a single video
def predict_video(video_path, model, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in np.linspace(0, frame_count - 1, num_frames, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (112, 112))
            frame = frame / 255.0
            frames.append(frame)
        else:
            frames.append(np.zeros((112, 112, 3)))

    cap.release()

    input_frames = np.expand_dims(np.array(frames), axis=0)
    prediction = model.predict(input_frames)
    return prediction[0][0]

# Example usage
print("Testing on a sample video...")
test_video_path = '/content/drive/MyDrive/test_video.mp4'
result = predict_video(test_video_path, lstm_model)
print(f"The video is classified as {'fake' if result > 0.5 else 'real'} with confidence {result:.2f}")

# Clean up temporary directory
print("Cleaning up...")
shutil.rmtree(temp_dir)

print("Deep fake detection model training and evaluation complete.")