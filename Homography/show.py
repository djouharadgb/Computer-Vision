import cv2
img = cv2.imread('djopics/pic1.jpg')
img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # resize to 1/4
cv2.imshow('image', img)
cv2.waitKey(0)