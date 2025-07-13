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

print("📸 Capturing every 2 seconds. Press 'q' to quit...")

last_capture_time = 0
interval = 2  # seconds

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
            # Analyze for emotion only
            analysis = DeepFace.analyze(
                frame,
                actions=['emotion'],
                enforce_detection=True,
                silent=True
            )

            emotion = analysis[0]['dominant_emotion']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_dir}/{timestamp}_{emotion}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Saved: {filename}")

        except Exception as e:
            print("⚠️ No face detected. Skipping capture.")

        last_capture_time = current_time

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
