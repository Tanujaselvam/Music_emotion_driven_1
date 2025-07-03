import os
import cv2
import time
import pygame
import numpy as np
import textwrap
from tensorflow.keras.models import load_model
from emotion_to_prompt import get_thought

# Paths
image_folder = "captured_images"
music_folder = "music"
emoji_folder = "emojis"

os.makedirs(image_folder, exist_ok=True)

pygame.mixer.init()

model = load_model('model.keras')
EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

BG_COLOR = (220, 222, 203)  # Soft tan/peach

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)
        preds = model.predict(roi, verbose=0)[0]
        emotion_idx = np.argmax(preds)
        emotion = EMOTIONS[emotion_idx]
        return emotion, (x, y, w, h)
    return None, None

def play_music(emotion):
    filename = f"{emotion}.mp3"
    music_file = os.path.join(music_folder, filename)
    if os.path.exists(music_file):
        pygame.mixer.music.load(music_file)
        pygame.mixer.music.play()
        print(f"Playing: {music_file}")
    else:
        print(f"Music file not found for emotion '{emotion}': {music_file}")

def overlay_emoji(frame, emoji_path, position, size=(70, 70)):
    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        return frame
    emoji = cv2.resize(emoji, size, interpolation=cv2.INTER_AREA)
    x, y = position
    h, w = emoji.shape[:2]
    if emoji.shape[2] == 4:
        alpha_s = emoji[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha_s * emoji[:, :, c] +
                                      alpha_l * frame[y:y+h, x:x+w, c])
    else:
        frame[y:y+h, x:x+w] = emoji
    return frame

def get_optimal_font_scale(text, width, font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
    """Find the maximum font scale that allows text to fit in the given width."""
    for scale in reversed(np.arange(0.3, 2.0, 0.05)):
        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        if text_size[0] <= width:
            return scale
    return 0.3

def display_full_thought(frame, thought):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img_h, img_w = frame.shape[:2]
    margin_x = 30
    margin_y = 60
    max_width = img_w - 2 * margin_x

    # Wrap text so each line fits the image width
    # First, guess a font scale
    font_scale = 0.8
    thickness = 3
    # Estimate average char width
    avg_char_width = cv2.getTextSize("A", font, font_scale, thickness)[0][0]
    chars_per_line = max_width // avg_char_width if avg_char_width else 40
    wrapped = textwrap.wrap(thought, width=int(chars_per_line))

    # Now, find the largest font scale that fits the widest line
    max_line = max(wrapped, key=len)
    font_scale = get_optimal_font_scale(max_line, max_width, font, thickness)

    # Draw each line
    y = margin_y
    for line in wrapped:
        cv2.putText(frame, line.strip(), (margin_x, y), font, font_scale, (185, 92, 185), thickness, cv2.LINE_AA)
        y += int(50 * font_scale) + 10

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot access camera.")
        return

    print("Press 'c' to capture, 'r' to resume webcam, 'q' to quit.")
    webcam_on = True

    while True:
        if webcam_on:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            cv2.imshow("Emotion Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            ret, captured_frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                continue

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(image_folder, f"{timestamp}.jpg")
            cv2.imwrite(image_path, captured_frame)
            print(f"Image saved: {image_path}")

            emotion, face_coords = detect_emotion(captured_frame)
            if emotion and face_coords:
                print(f"Detected emotion: {emotion}")
                thought = get_thought(emotion)
                print(f"Generated thought: {thought}")
                play_music(emotion)

                # Create pastel background
                h, w = captured_frame.shape[:2]
                bg = np.full((h, w, 3), BG_COLOR, dtype=np.uint8)

                # Paste enlarged face region in center
                x, y, fw, fh = face_coords
                # Enlarge face box by 1.5x, keep center
                scale = 1.5
                cx, cy = x + fw // 2, y + fh // 2
                new_fw, new_fh = int(fw * scale), int(fh * scale)
                nx = max(0, cx - new_fw // 2)
                ny = max(0, cy - new_fh // 2)
                nx2 = min(w, nx + new_fw)
                ny2 = min(h, ny + new_fh)
                face_img = captured_frame[ny:ny2, nx:nx2]
                # Center position
                face_h, face_w = face_img.shape[:2]
                bg_cx, bg_cy = w // 2 - face_w // 2, h // 2 - face_h // 2
                bg[bg_cy:bg_cy+face_h, bg_cx:bg_cx+face_w] = face_img

                # Overlay a single emoji in the bottom-right
                emoji_path = os.path.join(emoji_folder, f"{emotion}.png")
                emoji_size = (70, 70)
                ex, ey = w - emoji_size[0] - 25, h - emoji_size[1] - 25
                bg = overlay_emoji(bg, emoji_path, (ex, ey), emoji_size)

                # Display the thought at the top
                display_full_thought(bg, thought)

                # Show the result
                cv2.imshow("Emoji Burst Effect", bg)
                enhanced_path = os.path.join(image_folder, f"{timestamp}_enhanced.jpg")
                cv2.imwrite(enhanced_path, bg)
                webcam_on = False
            else:
                print("No face detected.")

        elif key == ord('r'):
            print("Resuming webcam...")
            webcam_on = True
            try:
                if cv2.getWindowProperty("Emoji Burst Effect", cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow("Emoji Burst Effect")
            except cv2.error:
                pass

        elif key == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
