import cv2
import cvlib
import numpy
from cvlib.object_detection import draw_bbox

while True:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap1 = cv2.VideoCapture('forward.jpg')
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    bourding_box, label, conf = cvlib.detect_common_objects(frame1)
    out = draw_bbox(frame, bourding_box, label,  conf ,write_conf=True)
    if frame.all()==frame1.all():
        print('forword')
    print(frame)
    cv2.imshow('img', out)
    cv2.waitKey(0)



