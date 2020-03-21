import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier(r'C:\Users\Chirag Arora\Desktop\Python-training\Lecture-32\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(r'C:\Users\Chirag Arora\Desktop\Python-training\Lecture-31\haarcascade_eye.xml')
glasses = cv2.imread(r"C:\Users\Chirag Arora\Desktop\glasses.png")

cap = cv2.VideoCapture(0)
	
while 1:
	retval, image = cap.read()
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face = face_cascade.detectMultiScale(gray)

	for (x,y,w,h) in face:
		cv2.rectangle(image, (x, y), (x+w, y+h), (0,0,255), 10)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = image[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0,255,0), 3)
			glasses = cv2.resize(glasses, (ew, eh))
		gx, gy, _ = glasses.shape
		for i in range(0, gx):
			for j in range(0, gy):
				if glasses[i, j][3] != 0:
				   image[gy+i, gx+j] = glasses[i, j]
			

	cv2.imshow('image', image)
	key = cv2.waitKey(30)
	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows() 