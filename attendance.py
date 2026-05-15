import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Haarcascade path
cascadePath = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Training data
path = 'images'

faces = []
ids = []
names = {}

imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

for idx, imagePath in enumerate(imagePaths):

    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    faces.append(img)
    ids.append(idx)

    name = os.path.split(imagePath)[-1].split(".")[0]
    names[idx] = name

recognizer.train(faces, np.array(ids))

# Webcam
cam = cv2.VideoCapture(0)

print("Smart Attendance System Started")

attendance_marked = []

while True:

    ret, frame = cam.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    facesDetected = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in facesDetected:

        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:

            name = names[id]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            cv2.putText(
                frame,
                name,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255,255,255),
                2
            )

            if name not in attendance_marked:

                now = datetime.now()
                time = now.strftime("%H:%M:%S")

                data = {
                    "Name": [name],
                    "Time": [time]
                }

                df = pd.DataFrame(data)

                if not os.path.isfile("attendance.csv"):
                    df.to_csv("attendance.csv", index=False)
                else:
                    df.to_csv(
                        "attendance.csv",
                        mode='a',
                        header=False,
                        index=False
                    )

                attendance_marked.append(name)

                print(f"Attendance Marked for {name}")

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
