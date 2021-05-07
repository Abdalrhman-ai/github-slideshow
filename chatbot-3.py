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
import wikipedia
import time

from time import sleep

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("datach.pickle", "rb") as f:
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

    with open("datach.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("modelch.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

r = sr.Recognizer()
m = sr.Microphone()
speaker = win32com.client.Dispatch("SAPI.SpVoice")
speaker.Speak("Hi, How can i help you ?")
print("Hi, How can i help you ?")
while True:
    with m as source: r.adjust_for_ambient_noise(source)
    print("Talk.....")
    with m as source: audio = r.listen(source)
    try:
        value = r.recognize_google(audio)
        if str is bytes: 
            print(format(value).encode("utf-8"))
        else: 
            print(format(value))
    except sr.RequestError as e:
        print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    except sr.UnknownValueError:
        #with m as source: r.adjust_for_ambient_noise(source)
        print("Talk.....")
        with m as source: audio = r.listen(source)
        try:
            value = r.recognize_google(audio)
            if str is bytes: 
                print(format(value).encode("utf-8"))
            else: 
                print(format(value))
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
            elif value=="open YouTube":
                with m as source: r.adjust_for_ambient_noise(source)
                speaker.Speak('What do you want from YouTube?: ')
                print('What do you want from YouTube?: ')
                with m as source: audio = r.listen(source)
                try:
                    value1 = r.recognize_google(audio)
                    if str is bytes: 
                        print(format(value).encode("utf-8"))
                    else: 
                        print(format(value))
                except sr.UnknownValueError:
                    break
            elif value=="open Google":
                speaker.Speak('What do you want from google')
                value2 = input('What do you want from google?: ')
                webbrowser.open('https://www.google.com/search?q='+ value2)
            elif value=="open Youtube":
                speaker.Speak('What do you want from YouTube')
                value1 = input('What do you want from YouTube?: ')
                webbrowser.open('https://www.youtube.com/results?search_query='+ value1)
            elif value=="what time is it":
                mytime = time.localtime()
                resultTime = time.strftime("%H:%M:%S", mytime)
                print("The time is now:",resultTime)
                speaker.Speak("The time is now: ")
                speaker.Speak(resultTime)
            elif value=="What is the date today":
                mytime2 = time.localtime()
                resultTime2 = time.strftime("%d/%m/%Y", mytime2)
                print("The date today is:", resultTime2 )
                speaker.Speak("The date today is: ")
                speaker.Speak(resultTime2)
            else:
                A=wikipedia.summary(value)
                print(A)
                speaker.Speak(A)
        except sr.RequestError as e:
             print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
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
    elif value=="open YouTube":
        with m as source: r.adjust_for_ambient_noise(source)
        speaker.Speak('What do you want from YouTube?: ')
        print('What do you want from YouTube?: ')
        with m as source: audio = r.listen(source)
        try:
            value1 = r.recognize_google(audio)
            if str is bytes: 
                print(format(value).encode("utf-8"))
            else: 
                print(format(value))
        except sr.UnknownValueError:
            break
    elif value=="open Google":
            speaker.Speak('What do you want from google')
            value2 = input('What do you want from google?: ')
            webbrowser.open('https://www.google.com/search?q='+ value2)
    elif value=="open Youtube":
           speaker.Speak('What do you want from YouTube')
           value1 = input('What do you want from YouTube?: ')
           webbrowser.open('https://www.youtube.com/results?search_query='+ value1)
    elif value=="what time is it":
            mytime = time.localtime()
            resultTime = time.strftime("%H:%M:%S", mytime)
            print("The time is now:",resultTime)
            speaker.Speak("The time is now: ")
            speaker.Speak(resultTime)
    elif value=="What is the date today":
         mytime2 = time.localtime()
         resultTime2 = time.strftime("%d/%m/%Y", mytime2)
         print("The date today is:", resultTime2 )
         speaker.Speak("The date today is: ")
         speaker.Speak(resultTime2)
    else:
        A=wikipedia.summary(value)
        print(A)
        speaker.Speak(A)

    