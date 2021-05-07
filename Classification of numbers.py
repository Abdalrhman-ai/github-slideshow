import tensorflow as tf
import numpy
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import tflearn 

(train_features, train_labels), (test_features, test_labels) = tf.keras.datasets.mnist.load_data()

train_features = tf.keras.utils.normalize(train_features, axis=1)
test_features = tf.keras.utils.normalize(test_features, axis=1)

model = Sequential()
net=model.add(tf.keras.layers.Flatten())
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
net=model.add(tf.keras.layers.Dense(50, activation = tf.nn.softmax))

model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_features, train_labels, epochs=3)

predictions = model.predict([test_features])

print("predictions value is: ", numpy.argmax(predictions[20]))

plt.imshow(test_features[20], cmap="gray")#cmap= plt.cm.binary)
plt.show()