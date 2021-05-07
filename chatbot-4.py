import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow as tf
import tflearn 
import random
import json
import pickle
import webbrowser
import win32com.client 
import pyttsx3
import speech_recognition as sr
from googlesearch import *
from time import sleep
import numpy as np
import face_recognition as fr
import cv2
import cvlib
from cvlib.object_detection import draw_bbox
from mtcnn import MTCNN

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("datachat.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = [] 
    for intent in data ["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)


        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("datachat.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("modelchat.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

bruno_image = fr.load_image_file("fase.jpg")
bruno_face_encoding = fr.face_encodings(bruno_image)[0]

known_face_encondings = [bruno_face_encoding]
known_face_names = ["Bruno"]

while True: 
    ret, frame = video_capture.read()

    bourding_box, label, conf = cvlib.detect_common_objects(frame)
    out = draw_bbox(frame, bourding_box, label,  conf ,write_conf=True)
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
        print(name)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    speaker.Speak("Hi, How can i help you ?")
    print("Hi, How can i help you ?")
    while True:
        r = sr.Recognizer()
        m = sr.Microphone()
        with m as source: r.adjust_for_ambient_noise(source)
        print("Recognizer...")
        with m as source: audio = r.listen(source)
        try:
            value = r.recognize_google(audio)
            if str is bytes: 
                print(format(value).encode("utf-8"))
            else: 
                print(format(value))
        except sr.UnknownValueError:
            print("You: ")
        except sr.RequestError as e:
            print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
   
        #if value.lower() == "quit":
                    #break
        results = model.predict([bag_of_words(value, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            A=speaker.Speak(Bot)
            print(Bot)
        elif value=="What is that":
            speaker.Speak("this is",label)
        else:
            speaker.Speak("This is the search results")
            webbrowser.open('https://www.google.com/search?q='+ value)


        cv2.imshow('Webcam_facerecognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_capture.release()
cv2.destroyAllWindows()