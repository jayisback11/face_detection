# Helped from Clever Programmer, https://www.youtube.com/watch?v=R7B8sSByZGQ&t=5957s&ab_channel=CleverProgrammer

import cv2

trained_front_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 0 means the main webcam you're using
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_front_face_data.detectMultiScale(grayscale_img)
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow("Face(s) Detection Webcam", frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()
