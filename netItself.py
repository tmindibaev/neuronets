# -*- coding: utf-8 -*-


import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard


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



X = train_genue
#X = tf.nn.l2_normalize(train_genue, 0, epsilon=1e-12, name=None)
X = np.vstack((train_genue, train_forged))
Y = np.concatenate((np.ones(train_genue.shape[0]), np.zeros(train_forged.shape[0])))
#Y = np.ones(train_genue.shape[0])

    
# create model
model = Sequential()
model.add(Dense(100, input_dim=13, activation="sigmoid", kernel_initializer="uniform"))
model.add(Dense(50, activation="sigmoid", kernel_initializer="uniform"))
model.add(Dense(25, activation="sigmoid", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
tensorboard = TensorBoard(log_dir=".\\output", 
                          histogram_freq=0, 
                          write_graph=True, 
                          write_images=True,
                          embeddings_freq=0)
model.fit(X, Y, epochs=100, batch_size=10, verbose=1, callbacks=[tensorboard])

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
