import cv2
import numpy as np
import pygame
from tensorflow.keras.models import load_model
import os

# Load the pre-trained emotion detection model
emotion_model = load_model('model.keras')

# Define emotion labels (these should match the model's output classes)
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize pygame for playing music
pygame.mixer.init()

# Function to play music based on detected emotion
def play_music(detected_emotion):
    music_folder = "music"
    # Capitalize the first letter to match your file names (e.g., Happy.mp3)
    filename = f"{detected_emotion.capitalize()}.mp3"
    music_file = os.path.join(music_folder, filename)
    if os.path.exists(music_file):
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        print(f"Playing: {music_file}")
    else:
        print(f"Music file not found for emotion '{detected_emotion}': {music_file}")

# Function to detect emotion from the captured frame
def detect_emotion(input_frame):
    # Convert the image to grayscale as the model expects
    gray_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model (only once in the script, it's not necessary to reload it each time)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

    # If no faces are detected, return None
    if len(faces) == 0:
        return None

    # For each face detected, process the first one (you can modify this logic to handle multiple faces)
    for (x, y, w, h) in faces:
        # Crop the face region of interest (ROI)
        roi_color = input_frame[y:y + h, x:x + w]

        # Convert the face ROI to grayscale
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

        # Resize the ROI to 48x48 pixels (the input size expected by the model)
        roi_resized = cv2.resize(roi_gray, (48, 48))

        # Normalize the image and add batch and channel dimensions
        roi_normalized = roi_resized.astype("float32") / 255.0
        roi_normalized = np.expand_dims(roi_normalized, axis=-1)  # Add channel dimension (48, 48, 1)
        roi_normalized = np.expand_dims(roi_normalized, axis=0)   # Add batch dimension (1, 48, 48, 1)

        # Predict the emotion probabilities
        emotion_probabilities = emotion_model.predict(roi_normalized)[0]

        # Get the emotion with the highest probability
        max_index = np.argmax(emotion_probabilities)
        predicted_emotion = emotion_labels[max_index]

        # Draw rectangle and display the emotion label on the face
        cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(input_frame, predicted_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return predicted_emotion

    # If no emotion was detected, return None
    return None

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set a flag to track whether we are capturing an image
capturing_image = False

while True:
    ret, video_frame = cap.read()

    if not ret:
        break

    # If capturing, detect emotion, display and play music
    if capturing_image:
        detected_emotion = detect_emotion(video_frame)

        if detected_emotion:
            # Display the detected emotion on the screen
            cv2.putText(video_frame, f"Detected Emotion: {detected_emotion}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(f"Detected Emotion: {detected_emotion}")

            # Play the corresponding music
            play_music(detected_emotion)

        capturing_image = False  # Stop capturing after one image

    # Display the video frame with emotion text
    cv2.imshow("Emotion Detection", video_frame)

    # Wait for key press events
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):  # Capture the image and detect emotion
        capturing_image = True
        print("Capturing image and playing music...")
        # Save the captured image
        cv2.imwrite(f"captured_image_{int(cv2.getTickCount())}.jpg", video_frame)

    elif key == ord('r'):  # Resume emotion detection
        print("Resuming emotion detection...")

    elif key == ord('q'):  # Quit the application
        print("Exiting...")
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
