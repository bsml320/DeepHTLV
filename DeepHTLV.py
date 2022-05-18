#!/usr/bin/env python
#DeepHTLV.py

from __future__ import print_function, division
import numpy as np
import h5py
import scipy.io
import random
import sys,os
import itertools
import numbers
from collections import Counter
from warnings import warn
from abc import ABCMeta, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

import tensorflow as tf

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(1337)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
#python_random.seed(1337)

# The below set_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/random/set_seed
#tf.random.set_seed(1337)
#older version of tensorflow
tf.set_random_seed(1337)

import os


# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3,4"



from keras.optimizers import RMSprop, SGD
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation, Flatten
import keras.layers.core as core
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge, multiply, Reshape
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.wrappers import Bidirectional
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Embedding
from sklearn.metrics import fbeta_score, roc_curve, auc, roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
from keras.regularizers import l2, l1, l1_l2
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import activations, initializers, regularizers, constraints
from keras.engine import InputSpec
from keras.layers import concatenate



# %%
class Attention(Layer):
	def __init__(self,hidden,init='glorot_uniform',activation='linear',W_regularizer=None,b_regularizer=None,W_constraint=None,**kwargs):
		self.init = initializers.get(init)
		self.activation = activations.get(activation)
		self.W_regularizer = regularizers.get(W_regularizer)
		self.b_regularizer = regularizers.get(b_regularizer)
		self.W_constraint = constraints.get(W_constraint)
		self.hidden=hidden
		super(Attention, self).__init__(**kwargs)
	    
	def build(self, input_shape):
		input_dim = input_shape[-1]
		self.input_length = input_shape[1]
		self.W0 = self.add_weight(name ='{}_W1'.format(self.name), shape = (input_dim, self.hidden), initializer = 'glorot_uniform', trainable=True) # Keras 2 API
		self.W  = self.add_weight( name ='{}_W'.format(self.name),  shape = (self.hidden, 1), initializer = 'glorot_uniform', trainable=True)
		self.b0 = K.zeros((self.hidden,), name='{}_b0'.format(self.name))
		self.b  = K.zeros((1,), name='{}_b'.format(self.name))
		self.trainable_weights = [self.W0,self.W,self.b,self.b0]
	    
		self.regularizers = []
		if self.W_regularizer:
			self.W_regularizer.set_param(self.W)
			self.regularizers.append(self.W_regularizer)
		
		if self.b_regularizer:
			self.b_regularizer.set_param(self.b)
			self.regularizers.append(self.b_regularizer)
	    
		self.constraints = {}
		if self.W_constraint:
			self.constraints[self.W0] = self.W_constraint
			self.constraints[self.W] = self.W_constraint
			
		super(Attention, self).build(input_shape)
	    
	def call(self,x,mask=None):
		attmap = self.activation(K.dot(x, self.W0)+self.b0)
		attmap = K.dot(attmap, self.W) + self.b
		attmap = K.reshape(attmap, (-1, self.input_length)) # Softmax needs one dimension
		attmap = K.softmax(attmap)
		dense_representation = K.batch_dot(attmap, x, axes=(1, 1))
		out = K.concatenate([dense_representation, attmap]) # Output the attention maps but do not pass it to the next layer by DIY flatten layer
		return out


	def compute_output_shape(self, input_shape):
	    return (input_shape[0], input_shape[-1] + input_shape[1])

	def get_config(self):
		config = {'init': 'glorot_uniform',
					'activation': self.activation.__name__,
					'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
					'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
					'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
					'hidden': self.hidden if self.hidden else None}
		base_config = super(Attention, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class attention_flatten(Layer): # Based on the source code of Keras flatten
	def __init__(self, keep_dim, **kwargs):
		self.keep_dim = keep_dim
		super(attention_flatten, self).__init__(**kwargs)

	def compute_output_shape(self, input_shape):
		if not all(input_shape[1:]):
			raise Exception('The shape of the input to "Flatten" '
							'is not fully defined '
							'(got ' + str(input_shape[1:]) + '. '
							'Make sure to pass a complete "input_shape" '
							'or "batch_input_shape" argument to the first '
							'layer in your model.')
		return (input_shape[0], self.keep_dim)   # Remove the attention map

	def call(self, x, mask=None):
		x=x[:,:self.keep_dim]
		return K.batch_flatten(x)




def build_model():
	print('building model')

	seq_input_shape = (1000,4)
	nb_filter = 256
	filter_length = 9
	attentionhidden = 256

	seq_input = Input(shape = seq_input_shape, name = 'seq_input')
	convul1   = Convolution1D(filters = nb_filter,
                        	  kernel_size = filter_length,
                        	  padding = 'valid',
                        	  activation = 'relu',
                        	  kernel_constraint = maxnorm(3),
                        	  subsample_length = 1)

	pool_ma1 = MaxPooling1D(pool_size = 3)
	dropout1 = Dropout(0.5977908689086315)
	dropout2 = Dropout(0.50131233477637737)
	decoder  = Attention(hidden = attentionhidden, activation = 'linear')
	dense1   = Dense(1)
	dense2   = Dense(1)

	output_1 = pool_ma1(convul1(seq_input))
	output_2 = dropout1(output_1)
	att_decoder  = decoder(output_2)
	output_3 = attention_flatten(output_2._keras_shape[2])(att_decoder)

	output_4 =  dense1(dropout2(Flatten()(output_2)))
	all_outp =  concatenate([output_3, output_4])
	output_5 =  dense2(all_outp)
	output_f =  Activation('sigmoid')(output_5)

	model = Model(inputs = seq_input, outputs = output_f)
	model.compile(loss = 'binary_crossentropy', optimizer = 'nadam', metrics = ['accuracy'])

	print (model.summary())
	return model



def run_model():

    x_visdb = np.load('data/x_VISDB_fulldata.npy')
    y_visdb = np.load('data/y_VISDB_fulldata.npy')

    trainx, valx, trainy, valy = train_test_split(x_visdb, y_visdb, test_size = 0.1, random_state = 42)

    model = build_model()
    model.load_weights('model/Final_model.h5')

    print('testing')
    
    y_pred = model.predict(valx, verbose = 1)

    auroc = roc_auc_score(valy, y_pred)
    aupr = average_precision_score(valy, y_pred)
    
    np.save('data/y_pred.npy', y_pred)
    np.save('data/valy.npy', valy)

    print('auroc = ', auroc)
    print('aupr = ', aupr)



if __name__ == '__main__':
	build_model()
	run_model()



