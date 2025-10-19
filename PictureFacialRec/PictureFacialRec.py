import os
from deepface import DeepFace
import cv2
from shutil import copy2

# Configurable paths
training_dir = "training_data"
input_images_dir = "input_images"
output_dir = "recognized_faces"
os.makedirs(output_dir, exist_ok=True)

# Only consider matches from these people
known_people = {"Person A"}

# Valid emotions to trigger save
valid_emotions = {"happy", "neutral", "surprise"}

# Step 1: Build face representations by triggering a find once
print("🧠 Building face representations from training data...")

# Just need to trigger one .find() to initialize the representations
sample_person = next(iter(known_people))
sample_path = os.path.join(training_dir, sample_person)
sample_image = next((f for f in os.listdir(sample_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.heic'))), None)

if not sample_image:
    raise ValueError(f"No images found in training_data/{sample_person}")

DeepFace.find(
    img_path=os.path.join(sample_path, sample_image),
    db_path=training_dir,
    model_name="Facenet",
    enforce_detection=False,
    silent=True
)

# Step 2: Match helper
def match_known_person(face_img_path):
    try:
        results = DeepFace.find(
            img_path=face_img_path,
            db_path=training_dir,
            model_name="Facenet",
            enforce_detection=False,
            silent=True
        )
        if len(results) > 0 and not results[0].empty:
            matched_path = results[0].iloc[0]['identity']  # get first matched path as string
            parts = os.path.normpath(matched_path).split(os.sep)
            person_name = parts[1] if len(parts) >= 2 else None
            if person_name in known_people:
                return person_name
    except Exception as e:
        print(f"⚠️ Matching error: {e}")
    return None

# Step 3: Process input images
print("🔍 Scanning input images...")
for filename in os.listdir(input_images_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(input_images_dir, filename)
    try:
        # Analyze faces & emotions
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion"],
            enforce_detection=True,
            silent=True
        )

        if not isinstance(analysis, list):
            analysis = [analysis]

        save_image = False
        matched_person = None

        for face in analysis:
            dominant_emotion = face["dominant_emotion"].lower()
            if dominant_emotion in valid_emotions:
                region = face["region"]
                img = cv2.imread(image_path)
                x, y, w, h = region["x"], region["y"], region["w"], region["h"]
                cropped_face = img[y:y+h, x:x+w]
                temp_face_path = "temp_face.jpg"
                cv2.imwrite(temp_face_path, cropped_face)

                matched_person = match_known_person(temp_face_path)
                if matched_person:
                    print(f"✅ Match: {matched_person} in {filename} with emotion: {dominant_emotion}")
                    save_image = True
                    break  # Save once per image if any match

        if save_image and matched_person:
            person_output_dir = os.path.join(output_dir, matched_person)
            os.makedirs(person_output_dir, exist_ok=True)
            copy2(image_path, os.path.join(person_output_dir, filename))
        else:
            print(f"❌ No valid match/emotion in {filename}")

    except Exception as e:
        print(f"⚠️ Error analyzing {filename}: {e}")
