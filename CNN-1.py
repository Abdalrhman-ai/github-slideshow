import tensorflow as tf
import numpy
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from random import shuffle
import numpy as np
import tflearn 
import os
import cv2

(train_features, train_labels), (test_features, test_labels) = datasets.cifar10.load_data()

train_features = tf.keras.utils.normalize(train_features, axis=1)
test_features = tf.keras.utils.normalize(test_features, axis=1)

train_features.shape
test_features.shape

train_labels.shape
train_labels[:5]

train_labels = train_labels.reshape(-1,)
train_labels[:5]

test_labels = test_labels.reshape(-1,)

model = Sequential()
net=model.add(tf.keras.layers.Flatten())
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(50, activation = tf.nn.softmax))

model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=5)

classes = ["airplane","car","bird","cat","deer","dog","frog","horse","ship","truck"]

model_out = model.predict(test_features)
model_out[:5]
y_classes = [np.argmax(element) for element in model_out]
y_classes[:5]
test_labels[:5]
label_out=classes[y_classes[100]]
print(label_out)
#print(y_classes)
def imshow(X, y, index):
    fig = plt.figure(figsize = (15,2))
    plt.imshow(X[index], cmap="gray")
    A=plt.xlabel(classes[y[index]])
    plt.title(label_out)
    print(A)
imshow(test_features, test_labels, 100)
plt.show()