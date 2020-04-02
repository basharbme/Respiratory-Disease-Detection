# -*- coding: utf-8 -*-
"""
Visualization of PCA vs autoencoder

Adapted from:
    https://stats.stackexchange.com/questions/190148/building-an-autoencoder-in-tensorflow-to-surpass-pca
Comments from: EMP
"""

# fyi - I had to manually install/upgrade many packages to get this part to run

from __future__ import absolute_import, division, print_function, unicode_literals

import random
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import layers, models



#%% create validation partition - doing at the top for consistency
# assuming 60,000 samples
train_size = 60000

# shrink the training set size to speed up the demo
# grab some training data to use as validation data
perc = 0.10
tr_ind = set(random.sample(list(np.arange(1,train_size)), int(np.floor(perc*train_size))))
val_ind = set(random.sample(list(np.arange(1,train_size)), int(np.floor(perc*train_size)))) - tr_ind

#%% now, load the mnist data
# handwritten digit data
# see: http://yann.lecun.com/exdb/mnist/

# load the testing and training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# break out training and validation
x_train= np.array(x_train)
x_val = x_train[list(val_ind)]
y_val = y_train[list(val_ind)]
x_train = x_train[list(tr_ind)]
y_train = y_train[list(tr_ind)]

# train a dense network to predict the digit

# reshape for the network
x_train = x_train.reshape(len(x_train), 28, 28, 1) / 255
x_val = x_val.reshape(len(x_val), 28, 28, 1) / 255
x_test = x_test.reshape(len(x_test), 28, 28, 1) / 255

#%% create our network!

# create a linear stack of layers
# see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
model = models.Sequential()
# the first layer is a convolutional layer:
#   - 32 filters (or kernels)
#   - kernel width of 3x3
#   - relu (rectified linear unit) activation
#   - input shape (from the data)
#   - see for more options: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), name="layer1"))
# then comes max pooling2
#   - argument describes how much to downsample the data (by a factor of 2 in both dimensions)
#   - see for more options: https://keras.io/layers/pooling/
model.add(layers.MaxPooling2D((2, 2), name="layer2"))
# then convolution again (same kernel size)
model.add(layers.Conv2D(64, (3,3), activation='relu', name="layer3"))
# then max pooling again
model.add(layers.MaxPooling2D((2, 2), name="layer4"))
# then another convolution layer
model.add(layers.Conv2D(64, (3,3), activation='relu', name="layer5"))

# flatten
model.add(layers.Flatten())
# dense
model.add(layers.Dense(64, activation='relu'))
# the final number of outputs need to match the number of classes (one for each class)
model.add(layers.Dense(10))

# compile the model
# using:
#   - the adam optimizer (see for more info: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
#   - categorical cross entropy (also know as: softmax!)
#   - accuracy as the metric (this works since the classes are relatively balanced)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#%% how many epochs (passes through the data)
num_epochs = 10

# this will hold the performance
perf_time = np.zeros((num_epochs, 3))

# set up figure
fig = plt.figure()
ax1 = fig.add_subplot(111)

# training the network for num_epoch epochs
for epoch in np.arange(0,num_epochs):
    # train an epoch at a time, visualize as we go!
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(x_val, y_val))
    
    # check the performance on train/test/val
    # the model.evaluate function returns the loss (position 0) and the performance (position 1)
    new = [model.evaluate(x_train, y_train)[1], model.evaluate(x_val, y_val)[1], model.evaluate(x_test, y_test)[1]]
    
    # add to performance
    perf_time[epoch,:]=new
    
    # visualize
    plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,0],'b', label='train')
    plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,1],'r', label='validation')
    plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,2],'g', label='test')
    plt.legend(loc='upper left')
    plt.show()
    
#%% repeat, using early stopping

# load the testing and training data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# break out training and validation
x_train= np.array(x_train)
x_val = x_train[list(val_ind)]
y_val = y_train[list(val_ind)]
x_train = x_train[list(tr_ind)]
y_train = y_train[list(tr_ind)]

# train a dense network to predict the digit

# reshape for the network
x_train = x_train.reshape(len(x_train), 28, 28, 1) / 255
x_val = x_val.reshape(len(x_val), 28, 28, 1) / 255
x_test = x_test.reshape(len(x_test), 28, 28, 1) / 255

# create a linear stack of layers
# see: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
model = models.Sequential()
# the first layer is a convolutional layer:
#   - 28 filters
#   - kernel width of 3x3
#   - relu (rectified linear unit) activation
#   - input shape (from the data)
#   - see for more options: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1), name="layer1"))
# then comes max pooling2
#   - argument describes how much to downsample the data (by a factor of 2 in both dimensions)
#   - see for more options: https://keras.io/layers/pooling/
model.add(layers.MaxPooling2D((2, 2), name="layer2"))
# then convolution again (same kernel size)
model.add(layers.Conv2D(64, (3,3), activation='relu', name="layer3"))
# then max pooling again
model.add(layers.MaxPooling2D((2, 2), name="layer4"))
# then another convolution layer
model.add(layers.Conv2D(64, (3,3), activation='relu', name="layer5"))

# flatten
model.add(layers.Flatten())
# dense
model.add(layers.Dense(64, activation='relu'))
# the final number of outputs need to match the number of classes (one for each class)
model.add(layers.Dense(10))

# compile the model
# using:
#   - the adam optimizer (see for more info: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)
#   - categorical cross entropy (also know as: softmax!)
#   - accuracy as the metric (this works since the classes are relatively balanced)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# how many epochs (passes through the data)
num_epochs = 1000

# this will hold the performance
perf_time = np.zeros((num_epochs, 4))

# set up figure
fig = plt.figure()
ax1 = fig.add_subplot(111)

# how well did it work?
best_val = [np.inf, 0]
for epoch in np.arange(0,num_epochs):
    # train an epoch at a time, visualize as we go!
    model.fit(x_train, y_train, batch_size=128, epochs=1, verbose=1, validation_data=(x_val, y_val))
    
    # check the performance on train/test/val
    # the model.evaluate function returns the loss (position 0) and the performance (position 1)
    val = model.evaluate(x_val, y_val)
    new = [model.evaluate(x_train, y_train)[1], 
           val[0], val[1], 
           model.evaluate(x_test, y_test)[1]]
    
    # add to performance
    perf_time[epoch,:]=new
    
    # visualize
    plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,0],'b', label='train')
    plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,2],'r', label='validation')
    plt.plot(np.arange(0,epoch+1),perf_time[0:epoch+1,3],'g', label='test')
    plt.legend(loc='upper left')
    plt.show()
    
    # is validation performance better?
    if val[0] >= best_val[0]:
        best_val[1] += 1
    else:
        best_val = [val[0], 0]
    print ("epoch %d, loss %f, number %d" %(epoch, best_val[0], best_val[1]))
        
    # if there hasn't been an improvement in three epochs, stop training
    if best_val[1] > 3:
        break
