import numpy as np
import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(r'C:\Users\Chirag Arora\Desktop\Python-training\Lecture-32\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r"C:\Users\Chirag Arora\Desktop\MachineLearning\haarcascade_eye.xml")
glasses = cv2.imread(r"C:\Users\Chirag Arora\Desktop\MachineLearning\single_eye.png")
	
while 1:
	retval, image = cap.read()
	if retval:
		faces = face_cascade.detectMultiScale(image)
		eyes = eye_cascade.detectMultiScale(image)
		for (x,y,w,h) in eyes:
			glasses = cv2.resize(glasses,(w, h))
			gx, gy, _ = glasses.shape
			for i in range(0, gx):
				for j in range(0, gy):
					if glasses[i, j][2]!= 0:
						image[y:y+w, x:x+w] = glasses[:,:]

		cv2.imshow('image',image)

	key = cv2.waitKey(30)
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows() 
