# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 12:48:16 2020

@author: prakh
"""

"""
Build an image classifier with Keras and Convolutional Neural Networks for the Fashion MNIST dataset.
This data set includes 10 labels of different clothing types with 28 by 28 grayscale images.
There is a training set of 60,000 images and 10,000 test images.

Label    Description
0        T-shirt/top
1        Trouser
2        Pullover
3        Dress
4        Coat
5        Sandal
6        Shirt
7        Sneaker
8        Bag
9        Ankle boot
"""

from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


plt.imshow(x_train[0])
y_train[0]

#Normalize the data using the max value
x_train.max()
x_test.max()
x_test=x_test/255
x_train=x_train/255

plt.imshow(x_train[0],cmap='gray')


#This is correct for a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel
#(since technically the images are in black and white, only showing values from 0-255 on a single channel),
#an color image would have 3 dimensions.
x_train.shape
x_test.shape

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

from keras.utils.np_utils import to_categorical
y_cat_test = to_categorical(y_test)
y_cat_train = to_categorical(y_train)


#Build the model
import tensorflow as tf
tf.__version__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten

#ORIGINAL
model = Sequential()
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))
# FLATTEN IMAGES FROM 28 by 28 to 764 BEFORE FINAL LAYER
model.add(Flatten())
# 128 NEURONS IN DENSE HIDDEN LAYER (YOU CAN CHANGE THIS NUMBER OF NEURONS)
model.add(Dense(128, activation='relu'))
# LAST LAYER IS THE CLASSIFIER, THUS 10 POSSIBLE CLASSES
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

model.fit(x_train,y_cat_train,epochs=10)

model.metrics_names

#Evalute on test - if its poor on test ang good on train- overfitting
model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_test)

y_cat_test.shape
predictions.shape

print(classification_report(y_test,predictions))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, predictions)
print(cm)
accuracy_score(y_test, predictions)




