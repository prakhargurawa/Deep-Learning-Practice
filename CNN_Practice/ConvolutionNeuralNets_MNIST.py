# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 18:58:37 2020

@author: prakh
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

single_image = x_train[0]
plt.imshow(single_image)
plt.imshow(single_image,cmap='gray')
print(single_image.shape)
print(y_train)

single_image = x_train[1]
plt.imshow(single_image)
plt.imshow(single_image,cmap='gray')

from keras.utils.np_utils import to_categorical
y_example = to_categorical(y_train)
print(y_example.shape)

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

x_train = x_train/255
x_test = x_test/255
scaled_single = x_train[0]
plt.imshow(scaled_single)


#This is correct for a CNN, but we need to add one more dimension to show we're dealing with 1 RGB channel
#(since technically the images are in black and white, only showing values from 0-255 on a single channel),
#an color image would have 3 dimensions.
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000,28,28,1)

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

"""
model = Sequential()
# CONVOLUTIONAL LAYER
model.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(28, 28, 1), activation='relu',))
# POOLING LAYER
model.add(MaxPool2D(pool_size=(2, 2)))
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
"""

model.summary()

model.fit(x_train,y_cat_train,epochs=2)

model.metrics_names
#Evalute on test - if its poor on test ang goof on train- overfitting
model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, predictions)
print(cm)
accuracy_score(y_test, predictions)
