
# 🎵 Emotion-Based Music Player

This project is an AI-powered **Emotion-Based Music Player** that detects a user's facial emotion through webcam input and plays music that matches the emotion. It combines computer vision, deep learning, and multimedia handling to create an adaptive, personalized music experience.

---

## 💡 Features

- 🎥 Real-time facial emotion detection using webcam
- 🧠 CNN-LSTM model for accurate emotion recognition
- 🎶 Emotion-matched music playback (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral)
- ⚡ Fast, local execution using OpenCV and TensorFlow
- 🔄 Automatically updates music if the user's emotion changes

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV** – for real-time webcam input and face detection
- **TensorFlow/Keras** – for CNN-LSTM model
- **NumPy, Pandas** – for data handling
- **Pygame / playsound / vlc** – for audio playback

---

## 🧠 Model Architecture

- **CNN** extracts spatial features from facial images
- **LSTM** captures temporal (time-based) patterns
- Together, the CNN-LSTM model provides reliable real-time emotion detection

---

## 🚀 How It Works

1. The webcam captures your face in real-time.
2. The face is processed and passed to a trained CNN-LSTM model.
3. The model predicts the emotion (e.g., Happy, Sad, Angry).
4. A music file matching the emotion is played from the local library.
5. If your emotion changes, the song changes too.

---

## 📁 Project Structure

