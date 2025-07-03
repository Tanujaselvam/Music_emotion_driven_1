import os
import cv2
import numpy as np

def show_emoji_burst(frame, emotion, face_coords):
    # Load the emoji image
    emoji_folder = "emojis"

    emoji_path = os.path.join(emoji_folder, f"{emotion}.png")
    if not os.path.exists(emoji_path):
        print(f"No emoji found for {emotion}")
        return frame

    emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
    if emoji is None:
        print(f"Failed to load emoji image for {emotion}")
        return frame

    # Create a copy of the frame
    result = frame.copy()

    # Add decorative border
    x, y, w, h = face_coords
    # Expand the region slightly
    padding = 50
    start_x = max(0, x - padding)
    start_y = max(0, y - padding)
    end_x = min(frame.shape[1], x + w + padding)
    end_y = min(frame.shape[0], y + h + padding)

    # Draw a rounded rectangle border
    cv2.rectangle(result, (start_x, start_y), (end_x, end_y), (255, 192, 203), 10)  # Pink border

    # Resize emoji to be smaller
    emoji_size = (40, 40)
    emoji_resized = cv2.resize(emoji, emoji_size, interpolation=cv2.INTER_AREA)

    # Add emojis around the face (fixed positions for more aesthetic placement)
    positions = [
        (start_x + 10, start_y + 10),
        (end_x - 50, start_y + 10),
        (start_x + 10, end_y - 50),
        (end_x - 50, end_y - 50),
        (start_x + (end_x - start_x) // 2 - 20, start_y - 10),
        (start_x - 10, start_y + (end_y - start_y) // 2 - 20),
        (end_x - 30, start_y + (end_y - start_y) // 2 - 20),
        (start_x + (end_x - start_x) // 2 - 20, end_y - 30)
    ]

    # Add additional random positions
    for _ in range(7):
        positions.append((
            np.random.randint(start_x + 20, end_x - 20),
            np.random.randint(start_y + 20, end_y - 20)
        ))

    # Place emojis at these positions
    for pos in positions:
        x, y = pos
        try:
            # Only place if within bounds
            if 0 <= x < result.shape[1] - emoji_size[0] and 0 <= y < result.shape[0] - emoji_size[1]:
                # For each pixel in the emoji
                for i in range(emoji_size[1]):
                    for j in range(emoji_size[0]):
                        if emoji_resized[i, j, 3] > 128:  # If not too transparent
                            result[y + i, x + j] = emoji_resized[i, j, :3]
        except IndexError:
            continue

    # Add sparkle effects
    sparkle_positions = [
        (start_x - 20, start_y - 20),
        (end_x + 10, start_y - 20),
        (start_x - 20, end_y + 10),
        (end_x + 10, end_y + 10)
    ]

    for pos in sparkle_positions:
        x, y = pos
        if 0 <= x < result.shape[1] - 20 and 0 <= y < result.shape[0] - 20:
            cv2.drawMarker(result, pos, (255, 192, 255), cv2.MARKER_STAR, 20, 2)

    return result
