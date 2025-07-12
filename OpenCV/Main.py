import cv2
import time
import os

# Set the camera source:
# 0 means the default webcam. Replace with your ESP32 video stream URL if needed.
CAMERA_SOURCE = 0

# Load the Haar Cascade for face detection.
# This file contains pre-trained data on how to detect a face.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the camera/video stream.
cap = cv2.VideoCapture(CAMERA_SOURCE)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Create a directory to store the cropped face images.
# It won't raise an error if the folder already exists.
save_dir = "captured_faces"
os.makedirs(save_dir, exist_ok=True)

# Counter for naming saved images
photo_counter = 1

print("Starting camera. Press 'q' to quit.")

# Main loop to read frames from the camera
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break  # If frame can't be read, exit the loop

    # Convert the captured frame to grayscale
    # This is required for the Haar Cascade face detection algorithm
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    # Parameters:
    #   scaleFactor = how much the image size is reduced at each image scale
    #   minNeighbors = how many neighbors each candidate rectangle should have to retain it
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        # Draw a green rectangle around the face in the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop just the face from the frame
        face_img = frame[y:y + h, x:x + w]

        # Define a unique filename for each detected face image
        filename = os.path.join(save_dir, f"party-guest_{photo_counter}.jpg")

        # Save the cropped face image to the directory
        cv2.imwrite(filename, face_img)

        # Log that the face was saved
        print(f"🎉 Face detected! Saved as {filename}")

        # Increment the counter for next face image
        photo_counter += 1

        # Add a short delay to avoid saving too many images of the same face
        time.sleep(2)

    # Display the live camera feed in a window
    cv2.imshow('Party Drone Cam', frame)

    # Wait for a key press for 1ms
    # If the user presses 'q', exit the loop and close the app
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
