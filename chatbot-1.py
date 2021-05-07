import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow as tf
import tflearn 
import random
import json
import webbrowser
import win32com
import win32com.client 
import pickle
import wikipedia
import time 

from time import sleep

with open("intents.json") as file: 
    data = json.load(file)

try:
    with open("datafromkaggel.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = [] 
    for intent in data ["intents"]:
        for pattern in intent["text"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["intent"])

        if intent["intent"] not in labels:
            labels.append(intent["intent"])

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
    output1 = numpy.array(output)

    with open("datafromkaggel.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("modelfromkaggel.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
   
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def chat():
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    print("Hello friend, happy to see you again")or("Hey friend")or("Hi friend how can i help you")
    speaker.Speak("Hello friend, happy to see you again")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        results_index1 = numpy.argmax(results)
        print(results_index,"and",results_index1)
        tag = labels[results_index]
        if results[results_index] > 0.8:
            for tg in data["intents"]:
                if tg['intent'] == tag:
                    responses = tg['responses']
            sleep(3)
            Bot = random.choice(responses)
            print(Bot)
            speaker.Speak(Bot)
        else:
            print("I do not understand. Please repeat the question, please")or("I do not understand")
            speaker.Speak("I do not understand. Please repeat the question, please")

chat()