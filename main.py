import numpy as np 
import pandas as pd
import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical
import tflearn.data_utils as du
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# read training & testing data

trainx = pd.read_csv("data/csvTrainImages 60k x 784.csv",header=None)
testx = pd.read_csv("data/csvTestImages 10k x 784.csv",header=None)
trainy = pd.read_csv("data/csvTrainLabel 60k x 1.csv",header=None)
testy = pd.read_csv("data/csvTestLabel 10k x 1.csv",header=None)

#Process data
trainx = trainx.values.astype('float32').reshape([-1, 28, 28, 1])
testx = testx.values.astype('float32').reshape([-1,28,28,1])

trainy = trainy.values.astype('int32')
trainy = to_categorical(trainy, 10)

testy = testy.values.astype('int32')
testy = to_categorical(testy, 10)

# Building convolutional network
network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 10, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')

#model complile
model = tflearn.DNN(network, tensorboard_verbose=0)

#model fitting
model.fit({'input': trainx}, {'target': trainy}, n_epoch=20,
           validation_set=({'input': testx}, {'target': testy}),
           snapshot_step=100, show_metric=True, run_id='convnet_arabic_digits')

# Evaluate model
score = model.evaluate(testx, testy)
print('Test accuarcy: %0.2f%%' % (score[0] * 100))