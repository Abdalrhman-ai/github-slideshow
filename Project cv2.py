import cv2
import cvlib
from cvlib.object_detection import draw_bbox
import time

while True:
	cap = cv2.VideoCapture("dog,cat.jpg")
	ret, frame = cap.read()
	bourding_box, label, conf = cvlib.detect_common_objects(frame)
	out = draw_bbox(frame, bourding_box, label, conf, write_conf=True)
	cv2.imshow('out',out)
	cv2.waitKey(0)
	if label==['stop sign']:
		print('stop')
		time.sleep(5)
		print('go')