# Helped from Clever Programmer, https://www.youtube.com/watch?v=R7B8sSByZGQ&t=5957s&ab_channel=CleverProgrammer

import cv2

# Below is the trained data which we got from HaarCascade website.
trained_front_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('xhio.png')

# Convert the img from color to grayscale because the trained data only works best on grayscale.
img_grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_front_face_data.detectMultiScale(img_grayscaled)

for (x,y,w,h) in face_coordinates:
    #draw the square
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 225, 0), 2)

cv2.imshow('Face(s) Detection', img)
cv2.waitKey()

