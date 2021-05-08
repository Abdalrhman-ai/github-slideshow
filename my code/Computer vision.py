from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2 as cv
import math
import time
import argparse
import numpy as np
import face_recognition as fr
import win32com
import win32com.client 
import cv2
import sys

t = time.time()
video_capture = cv2.VideoCapture("dog,cat.jpg")

bruno_image = fr.load_image_file("fase.jpg")
bruno_face_encoding = fr.face_encodings(bruno_image)[0]

known_face_encondings = [bruno_face_encoding]
known_face_names = ["Elon Mask"]

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument("-i", help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "modelNweight/opencv_face_detector.pbtxt"
faceModel = "modelNweight/opencv_face_detector_uint8.pb"

ageProto = "modelNweight/age_deploy.prototxt"
ageModel = "modelNweight/age_net.caffemodel"

genderProto = "modelNweight/gender_deploy.prototxt"
genderModel = "modelNweight/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(From 0 to 2)', '(From 4 to 6)', '(From 8 to 12)', '(From 15 to 20)', '(From 25 to 32)', '(From 38 to 43)', '(From 48 to 53)', '(From 60 to 100)']
genderList = ['Male','Female']

ageNet = cv.dnn.readNetFromCaffe(ageProto,ageModel)
genderNet = cv.dnn.readNetFromCaffe(genderProto,genderModel)
faceNet = cv.dnn.readNet(faceModel,faceProto)

padding = 20

ret, frame = video_capture.read()
frameFace, bboxes = getFaceBox(faceNet, frame)

for bbox in bboxes:
     face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

     blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
     genderNet.setInput(blob)
     genderPreds = genderNet.forward()
     gender = genderList[genderPreds[0].argmax()]
        

     ageNet.setInput(blob)
     agePreds = ageNet.forward()
     age = ageList[agePreds[0].argmax()]
        

     label = "{},{}".format(gender, age)
     #cv.putText(frame, label, (bbox[0]-5, bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0,255), 2, cv.LINE_AA)
     name = args.i

small_frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
rgb_frame = small_frame[:,:,::-1]
face_locations = fr.face_locations(rgb_frame)
face_encodings = fr.face_encodings(rgb_frame, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encondings, face_encoding)
        name = "Unknown person"

        face_distances = fr.face_distance(known_face_encondings, face_encoding)

        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        #cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]


faces = face_detection.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
if len(faces) > 0:
    faces = sorted(faces, reverse=True,key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
    (fX, fY, fW, fH) = faces
    roi = gray[fY:fY + fH, fX:fX + fW]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
        
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    #cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    #cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),(0, 0, 255), 2)
speaker = win32com.client.Dispatch("SAPI.SpVoice")
print("-------------------------------------------------------------------------")
print("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))
print("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))
print(name)
print(label)
print("Time : {:.3f}".format(time.time() - t))
print("-------------------------------------------------------------------------")
#speaker.Speak("Gender : {}, confidence = {:.3f}".format(gender, genderPreds[0].max()))
#speaker.Speak("Age : {}, confidence = {:.3f}".format(age, agePreds[0].max()))
#speaker.Speak(name)
#speaker.Speak(label)
#speaker.Speak("Time : {:.3f}".format(time.time() - t))
while True:
    cv.imshow("output", frame)        
    cv2.waitKey(0)
    video_capture.release()
cv2.destroyAllWindows()
