# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 11:58:17 2018

@author: rahul
"""

###Convulated Neural Networks: Identifying dogs and cats from images

##importing libraries
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

##Designing the CNN
##initializing the CNN

classifier = Sequential()
##adding the convolution layer
classifier.add(Convolution2D(32,3,3, input_shape = (64, 64, 3), activation = "relu"))

##adding the pooling step
classifier.add(MaxPooling2D(pool_size = (2,2), strides = 2))


###Adding additional Convolutional layer
classifier.add(Convolution2D(32,3,3, activation = "relu"))

classifier.add(MaxPooling2D(pool_size = (2,2), strides = 2))


###adding the flattening step
classifier.add(Flatten())

###adding the fully connected layer(hidden)
classifier.add(Dense(output_dim = 128, activation = "relu"))
##adding the output layer
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))

##compiling the CNN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])



###Image data generator
from keras.preprocessing.image import ImageDataGenerator
#import PIL
#from PIL import Image

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_generator = test_datagen.flow_from_directory('dataset/test_set',
                                                  target_size=(64,64),
                                                  batch_size=32,class_mode='binary')

classifier.fit_generator(train_generator,steps_per_epoch=8000,epochs=25,validation_data=test_generator,validation_steps=2000)

