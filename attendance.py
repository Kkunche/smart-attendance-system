import cv2
import os
import numpy as np
from datetime import datetime

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Load face detector
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

path = 'images'

faces = []
ids = []
names = {}

imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

# Training images
for idx, imagePath in enumerate(imagePaths):

    grayImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    faces.append(grayImg)

    ids.append(idx)

    name = os.path.split(imagePath)[-1].split(".")[0]

    names[idx] = name

recognizer.train(faces, np.array(ids))

# Create attendance file if not exists
if not os.path.exists("attendance.csv"):

    with open("attendance.csv", "w") as f:
        f.write("Name,Time\n")

cam = cv2.VideoCapture(0)

marked_names = []

print("Attendance System Started")

while True:

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detectedFaces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in detectedFaces:

        id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        if confidence < 70:

            name = names[id]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )

            if name not in marked_names:

                now = datetime.now()

                current_time = now.strftime("%H:%M:%S")

                with open("attendance.csv", "a") as f:

                    f.write(f"{name},{current_time}\n")

                marked_names.append(name)

                print(f"Attendance Marked: {name}")

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()