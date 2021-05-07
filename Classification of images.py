import cv2
import os
import numpy as np
from random import shuffle
import tensorflow as tf
import tflearn 
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import tflearn 

def label_img(img):
    word_label = img.split('_')[0000]
    if word_label=="flower": return[0,1]
    elif word_label=="fruit": return[1,0]
    elif word_label=="cat": return[1,1]
    elif word_label=="dog": return[0,0]
    elif word_label=="airplane": return[0,2]
    elif word_label=="car": return[1,2]
    elif word_label=="motorbike": return[2,2]
    elif word_label=="person": return[0,3]
IMG_SIZE = 120

def trining_Date():
    TRAIN_DTR = 'Negro dynasty/natural_images/train_1'
    trining_date = []
    for img in os.listdir(TRAIN_DTR):
        label = label_img(img)
        path = os.path.join(TRAIN_DTR, img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        trining_date.append([img, label])
    shuffle(trining_date)
    return trining_date
def test_Date():
    TEST_DTR = 'Negro dynasty/natural_images/test_1'
    test_date = []
    for img in os.listdir(TEST_DTR):
        label = label_img(img)
        path = os.path.join(TEST_DTR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
        test_date.append([img, img_num])
    shuffle(test_date)
    return test_date

trining_date = trining_Date()
test_date = test_Date()

net = tflearn.input_data(shape=[None , IMG_SIZE, IMG_SIZE, 1])
net = tflearn.conv_2d(net, 32, 5, activation = "relu")
net = tflearn.max_pool_2d(net, 5)
net = tflearn.conv_2d(net, 64, 5, activation = "relu")
net = tflearn.max_pool_2d(net, 5)
net = tflearn.conv_2d(net, 128, 5, activation = "relu")
net = tflearn.max_pool_2d(net, 5)
net = tflearn.conv_2d(net, 64, 5, activation = "relu")
net = tflearn.max_pool_2d(net, 5)
net = tflearn.conv_2d(net, 32, 5, activation = "relu")
net = tflearn.max_pool_2d(net, 5)

net = tflearn.fully_connected(net, 1024, activation = "relu")
net = tflearn.dropout(net , 0.8)
net = tflearn.fully_connected(net, 2, activation = "softmax")

net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                 loss='mean_square', name='targets')

model = tflearn.DNN(net)

training = trining_date[-1000:]
validation = trining_date[:-1000]

training_features = np.array([i[0] for i in training]).reshape(-1,IMG_SIZE,IMG_SIZE, 1)
training_labels = [i[1] for i in training]

validation_features = np.array([i[0] for i in validation]).reshape(-1,IMG_SIZE,IMG_SIZE, 1)
validation_labels = [i[1] for i in validation]

model.fit(training_features , training_labels, n_epoch=5 ,validation_set=(validation_features, validation_labels))

fig = plt.figure()
for num,date in enumerate(test_date[:1]):
    img_date = date[0]
    y = fig.add_subplot(3,4,num+1)
    date = img_date.reshape(IMG_SIZE,IMG_SIZE, 1)
    model_out = model.predict([date])[0]
    print(np.argmax(model_out))
    if np.argmax(model_out) == 1:str_label="fruit"
    elif np.argmax(model_out) == 0:str_label="flower"
    else: str_label="I do not know"
    plt.title(str_label)
    y.imshow(img_date, cmap="gray")
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()
