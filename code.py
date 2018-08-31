# -*- coding: utf-8 -*-
"""
Advanced Signal Processing Laboratory
Image Recognition Assignment
Katja Karsikas

The task is to categorize images of cats and dogs into two classes.
First, the data is preprocessed and then used in two different kind of convolutional neural networks to solve the
classification problem. The two networks are a smaller network built from scratch according to a topology
given in assignment instructions and a larger fine-tuned pretrained VGG16 network.
"""

import glob
import numpy as np
import xml.etree.ElementTree as ET
import cv2
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.layers import Input
from keras.optimizers import SGD
from keras.utils import to_categorical
import matplotlib.pyplot as plt


#  Extract the data and parse the cat/dog labels
name_list = glob.glob('annotations\\xmls\\*.xml')
data = []
labels = []

for name in name_list:
    img_path = 'images\\' + name + '.jpg'
    ann_path = 'annotations\\xmls\\' + name
    
    # Parse xml file    
    tree = ET.parse(name)
    root = tree.getroot()
    c = root[5][0].text # Class (cat or dog)
    if c == "cat":
        label = 0
    elif c == "dog":
        label = 1
    xmin = int(root[5][4][0].text)
    ymin = int(root[5][4][1].text)
    xmax = int(root[5][4][2].text)
    ymax = int(root[5][4][3].text)
    filename = root[1].text  
    
    # Read the image using cv2
    img_path = 'images\\' + filename
    image = cv2.imread(img_path)
     
    '''
    cv2.imshow('Test image',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''

    # Preprocess and resize the image
    image = image.astype('float32')
    image /= 255 # normalize pixel values to the range 0-1
    box = image[ ymin:ymax, xmin:xmax, : ] # Croping a square shaped bounding box around the head  
    resized_image = zoom(box, (64/box.shape[0], 64/box.shape[1], 1)) #The bounding box is resized to fixed size 64x64 with three color channels 
    
    '''
    cv2.imshow('Resized image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    # Add image and class label to arrays
    data.append(resized_image)
    labels.append(label)

# Change arrays to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split data to train and test data
(x_train, x_test, y_train, y_test) = train_test_split(data, labels, test_size=0.2, random_state=42)


### CNN from scratch

model = Sequential()
num_featmaps = 32
num_classes = 2
w, h = 3,3 # conv window size

# Topology of the network
# Layer 1: Input layer
model.add(Convolution2D(num_featmaps, kernel_size=(w, h), padding='same', activation='relu', input_shape=(64,64,3)))
# Layer 2: Conv2D + MaxPooling
model.add(Convolution2D(num_featmaps, kernel_size=(w, h), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Layer 3: Conv2D
model.add(Convolution2D(num_featmaps, kernel_size=(w, h), padding='same', activation='relu'))
# Layer 4: Conv2D + MaxPooling
model.add(Convolution2D(num_featmaps, kernel_size=(w, h), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Layer 5: Conv2D
model.add(Convolution2D(num_featmaps, kernel_size=(w, h), padding='same', activation='relu'))
# Layer 6: Conv2D + MaxPooling
model.add(Convolution2D(num_featmaps, kernel_size=(w, h), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# The fully-connected layers 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Create a model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

# Train the model
history = model.fit(x_train, y_train, batch_size=128, epochs=50, verbose=1, validation_data=(x_test, y_test), shuffle=True)

# Print the structure of model
model.summary()

# Evaluate the trained model on the test set
score = model.evaluate(x_test, y_test, verbose=1) 
print('Test accuracy:', score[1])

# Draw training accuracy and loss plots
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


###  A fine-tuned network based on a large pretrained one

# Load VGG16 pretrained network
vgg16 = VGG16(weights='imagenet', include_top=False)
# Print the structure of loaded pretrained model
vgg16.summary()

# Freeze the weights of the layers of the pre-trained network
for layer in vgg16.layers:
    layer.trainable = False

# Create own input format
keras_input = Input(shape=(64, 64, 3))

output_vgg16 = vgg16(keras_input)

# Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16)
x = Dense(4096, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
x = Dense(4096, activation='relu', kernel_initializer='random_uniform', bias_initializer='zeros')(x)
x = Dense(num_classes, activation='softmax', kernel_initializer='random_uniform', bias_initializer='zeros')(x)

# Create a model 
vgg16_model = Model(inputs=keras_input, outputs=x)
vgg16_model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.0001, momentum=0.9), metrics=['accuracy'])     

# Print the structure of modified pretrained model
vgg16_model.summary()

# Change class label data to categorial
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

# Train the model
history = vgg16_model.fit(x_train, y_train_cat, batch_size=128, epochs=50, verbose=1, validation_data=(x_test, y_test_cat), shuffle=True)

# Evaluate the trained model
score = vgg16_model.evaluate(x_test, y_test_cat, verbose=1)
print('Test accuracy:', score[1])

# Draw training accuracy and loss plots
# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
