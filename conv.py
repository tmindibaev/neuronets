# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:20:33 2017

@author: timofey
"""

# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D

train_genue = np.genfromtxt('train.csv', delimiter=',')
train_forged = np.genfromtxt('train_f.csv', delimiter=',')
test_genue = np.genfromtxt('test.csv', delimiter=',')
test_forged = np.genfromtxt('test_f.csv', delimiter=',')

def normalization(arr):
    for i in range(0, arr.shape[1]):
        #minimum = arr[:, i].min()
        maximum = arr[:, i].max()
        if maximum == 0:
            arr[:, i] = 0
        else:
            for j in range(0, arr.shape[0]):
                arr[j, i] = arr[j, i]/maximum
    return arr

train_genue = normalization(train_genue)
train_forged = normalization(train_forged)
test_genue = normalization(test_genue)
test_forged = normalization(test_forged)

#train_genue = tf.placeholder(tf.float32, (128, 13, 1))
#train_forged = tf.placeholder(tf.float32, (254, 13, 1))
#


X = train_genue
#X = tf.nn.l2_normalize(train_genue, 0, epsilon=1e-12, name=None)
X = np.vstack((train_genue, train_forged))
Y = np.concatenate((np.ones(train_genue.shape[0]), np.zeros(train_forged.shape[0])))
#Y = np.ones(train_genue.shape[0])
#Y = tf.placeholder(tf.float32, (382, 1, 1))
X = np.reshape(X,newshape=[254, 13, 1], order = 'C')
test_forged = np.reshape(test_forged, newshape= (36, 13, 1), order = 'C')
test_genue = np.reshape(test_genue, newshape = (128, 13, 1), order = 'C')
Y = np.reshape(Y, newshape = (254, 1, 1), order = 'C')

# create model


model = Sequential()

model.add(Conv1D(strides=10,
                 filters=1,
                 input_shape=(None, 1),
                 kernel_size=3,
                 activation="tanh", 
                 kernel_initializer='glorot_uniform'))      
model.add(Conv1D(strides=1,
                 filters=1,
                 kernel_size=1,
                 activation="tanh", 
                 kernel_initializer='uniform'))  


#model.add(Dense(100, input_dim=13, activation="tanh", kernel_initializer="uniform"))
#model.add(Dense(25, activation="tanh", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
tensorboard = TensorBoard(log_dir=".\\output", histogram_freq=0, write_graph=True, write_images=True)


model.fit(X, Y, epochs=2000, batch_size=200, verbose=1, callbacks=[tensorboard])
# calculate predictions
predictions = model.predict(X)
predictions1 = model.predict(test_forged)
predictions2 = model.predict(test_genue)
# round predictions
#rounded = [round(x[0]) for x in predictions]
#rounded1 = [round(x[0]) for x in predictions1]
#rounded2 = [round(x[0]) for x in predictions2]
'''predictions = Y #- rounded
predictions1 = np.zeros(test_forged.shape[0]) #- rounded1
predictions2 = np.ones(test_genue.shape[0]) #- rounded2'''
print(predictions)
print(predictions1)
print(predictions2)

'''for layer in model.layers:
    weights = layer.get_weights()'''
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')
#model.summary()
