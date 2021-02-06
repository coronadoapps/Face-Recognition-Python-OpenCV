import cv2
import numpy as np
import dlib

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Loading Camera and Nose image and Creating mask
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    _, frame = cap.read()

    #Processing frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray)
    faceMask = np.zeros_like(gray)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        landmarks = predictor(gray, face)
        landmarks_points = []

        for n in np.arange(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x,y))

            cv2.circle(frame, (x, y), 2, (255,255,255), 1)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()