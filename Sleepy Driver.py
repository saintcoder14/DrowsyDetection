# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 00:51:21 2020
@author: Samridh
"""

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import playsound
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
	playsound.playsound(path)

def EAR(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

eye_threshold=0.3
frames_limit=50
count=0

print("[INFO] loading facial landmark predictor...")

detector = cv2.CascadeClassifier(r"C:\Users\samri\Desktop\Project\Sleepy Driver Alert Detection [Python, OpenCV]\Resources\haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor(r"C:\Users\samri\Desktop\Project\Sleepy Driver Alert Detection [Python, OpenCV]\Resources\shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] starting video stream thread...")

vs = VideoStream(src=0).start()
time.sleep(1.0)

while True:
# Frame is then preprocessed by resizing it to have a width of '450 pixels' and converting it to grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# dlib’s face detector to find and locate the face(s) in the image.
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	for (x, y, w, h) in rects:
		rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
		shape= predictor(gray, rect)
		shape= face_utils.shape_to_np(shape)

		leftEye= shape[lStart:lEnd]
		rightEye= shape[rStart:rEnd]
		leftEAR= EAR(leftEye)
		rightEAR= EAR(rightEye)

		ear=(leftEAR+rightEAR)/2.0

		leftEyeHull= cv2.convexHull(leftEye)
		rightEyeHull= cv2.convexHull(rightEye)

		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear<eye_threshold:
			count+=1

			if count>=frames_limit:

				sound_alarm(r"C:\Users\samri\Desktop\Project\Sleepy Driver Alert Detection [Python, OpenCV]\alarm.wav")
				cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		else:
			count=0

		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()