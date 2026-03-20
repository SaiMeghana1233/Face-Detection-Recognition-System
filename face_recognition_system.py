import cv2
import face_recognition
import os

# -----------------------------
# 1️⃣ Load known faces
# -----------------------------
known_face_encodings = []
known_face_names = []

known_faces_folder = "known_faces"

for filename in os.listdir(known_faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(known_faces_folder, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Only add if a face is found
            known_face_encodings.append(encodings[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Use filename as name
        else:
            print(f"No face found in {filename}, skipping.")

print(f"Loaded {len(known_face_encodings)} known faces.")

# -----------------------------
# 2️⃣ Start webcam
# -----------------------------
video_capture = cv2.VideoCapture(0)

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

    # -----------------------------
    # 3️⃣ Detect faces
    # -----------------------------
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    recognized_in_frame = []  # To avoid printing same name multiple times per frame

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            if name not in recognized_in_frame:
                print(f"Recognized: {name}")
                recognized_in_frame.append(name)

        # Draw rectangle and name on webcam feed
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # -----------------------------
    # 4️⃣ Display webcam feed
    # -----------------------------
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# 5️⃣ Cleanup
# -----------------------------
video_capture.release()
cv2.destroyAllWindows()
print("Webcam closed. Program ended.")