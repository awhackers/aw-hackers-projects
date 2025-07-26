import cv2
import os
import time
from datetime import datetime
from deepface import DeepFace

# Create directory to save snapshots
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("📸 Capturing every 5 seconds. Press 'q' to quit...")

last_capture_time = 0
interval = 5  # seconds
valid_emotions = {"happy", "neutral", "surprise"}

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Display live video feed
    cv2.imshow("Emotion Capture (press 'q' to quit)", frame)

    current_time = time.time()
    if current_time - last_capture_time >= interval:
        try:
            # Analyze all detected faces
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=True,
                silent=True
            )

            # Make sure it's a list of faces
            if not isinstance(analysis, list):
                analysis = [analysis]

            # Check if any face matches the valid emotions
            should_save = any(
                face["dominant_emotion"].lower() in valid_emotions for face in analysis
            )

            if should_save:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_dir}/{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✅ Saved: {filename}")
            else:
                print("ℹ️ No matching emotions found. Skipping.")

        except Exception as e:
            print(f"⚠️ No face detected or error: {str(e)}")

        last_capture_time = current_time

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
