#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json
import os
import cv2
import skvideo.io
import numpy as np
import matplotlib.pyplot as plt
import c3d_model
import sys
import keras.backend as K
dim_ordering = K.image_dim_ordering()
backend = dim_ordering

import warnings
import threading
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from functools import partial
import itertools

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, merge, Lambda
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import *
from keras.regularizers import l2,l1
from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.utils import np_utils
from keras import optimizers
from keras.activations import relu
from keras.utils import np_utils
from keras.optimizers import RMSprop,SGD,Adam
from keras.layers.merge import add,concatenate

from demo_algorithm import key_frame
warnings.filterwarnings("ignore")

machine ='ye_home'

# Path to the directories of features and labels
if machine == 'ye_home':
	path = '/home/ye/Works'
elif machine == 'MARCC':
	path = '/home-4/ytian27@jhu.edu/scratch/yetian'

# frames_dir = os.path.join(path, 'C3D-TCN-Keras/frames')
frames_dir = os.path.join(path, 'C3D-TCN-Keras/frames')
label_dir = os.path.join(path, 'C3D-TCN-Keras/classInd.txt')


def raw_model(n_classes):
	model_dir = './models'
	model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
	model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

	print("[Info] Reading model architecture...")
	# model = model_from_json(open(model_json_filename, 'r').read())
	model = c3d_model.get_model(backend=backend)

	print("[Info] Loading model weights...")
	model.load_weights(model_weight_filename)
	model.add(Dense(n_classes, activation='softmax'))

	print("[Info] Loading model weights -- DONE!")
	model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])
	return model


######################################################################################################
def trans_label(txt):
	# label_list = np.loadtxt(txt)
	label_list = np.genfromtxt(txt, delimiter=' ', dtype=None)
	label_dict = {}
	# num_lab = len(label_list)
	num_lab = 101
	for i in range(num_lab):
		# print label_list[i][1], label_list[i][0]
		label_dict[label_list[i][1]] = label_list[i][0]
	# print label_dict['HandStandPushups']
	return label_dict

#####################################################################################################
def to_vector(mat):
	"""Convert categorical data into vector.
	Args:
		mat: onr-hot categorical data.
	Returns:
		out2: vectorized data."""
	out = np.zeros((mat.shape[0],mat.shape[1]))
	out2 = np.zeros((mat.shape[0]))
	for i in range(mat.shape[0]):
		for n, j in enumerate(mat[i]):
			if j == np.amax(mat[i]):
				out[i][n] = 1
				out2[i] = n

	return out2
#####################################################################################################
class Iterator(object):
	def __init__(self, N, batch_size, shuffle, seed):
		self.N = N
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.batch_index = 0
		self.total_batches_seen = 0

	def __iter__(self):
		return self

	def __next__(self, *args, **kwargs):
		return self.next(*args, **kwargs)

######################################################################################################
class VideoIterator(Iterator):
	def __init__(
		self, 
		data_generator, 
		video_files,
		nb_classes,
		cache_dir,
		batch_size, 
		keyframe,
		shuffle=False, 
		seed=None):

		self.data_generator = data_generator
		self.video_files = video_files
		self.nVideos = len(self.video_files)

		self.cache_dir = cache_dir

		self.max_sequence_len = 16
		self.nb_classes = nb_classes
		self.batch_size = batch_size

		self.keyframe = keyframe

		self.vid_size_cache = {}
		self.label_dict = trans_label(label_dir)

		# The input sequence consists of concatenated video clips, up to a maximum length of self.max_sequence_len:
		self.Xtrain = np.zeros((self.batch_size,self.max_sequence_len,112,112,3), dtype=np.float32)

		# The output is a frame-level segmentation (one-hot labeling, per-frame for the entire sequence)
		self.ytrain = np.zeros((self.batch_size,self.nb_classes), dtype=np.float32)

		self.ji = 0

	def next(self):
		"""
		do the work here
		"""

		# Initialize the input sequence batch of features to all 0's initially
		self.Xtrain[:] = 0.0

		# Initialize the output sequence batch of labels to all 0's initially
		self.ytrain[:] = 0.0

		# Construct a batch to update the training
		b = 0

		ji = self.ji

		while b < self.batch_size:

			start = 0
			break_out = False

			vid_file0 = self.video_files[ji]
			vid_folder = os.path.join(self.cache_dir,vid_file0)
			# print vid_folder
			n_frames0 = len(os.listdir(vid_folder))

			if n_frames0 >= self.max_sequence_len:
				if ji > self.nVideos-1:
					self.ji = 0
					ji = 0
				# end = start + n_frames0
				
					# end = self.max_sequence_len
					# n_frames0 = end-start
					# break_out = True
				if self.keyframe == False:
					img_names = sorted(os.listdir(vid_folder))[:self.max_sequence_len-1]
				else:
					img_names = key_frame(vid_folder, 5, self.max_sequence_len)


				# Retrieve the cached video features 
				label0 = self.label_dict[vid_file0[2:-8]] -1
				# print img_names[0]
				for i in range(len(img_names)):
					img = cv2.imread(os.path.join(vid_folder,img_names[i]))
					# print img.shape
					self.Xtrain[b,i] = cv2.resize(img,(112,112))

				# Store the labels for these frames
				self.ytrain[b,label0] = 1.0   

				# start = end  # update for next sequence   
				# And increment the batch index
				b += 1
			ji += 1

		self.ji = ji

		return self.Xtrain, self.ytrain



####################################################################################################
class MyDataGenerator(object):
	def __init__(self, video_files, nb_classes, cache_dir, keyframe):
		self.video_files = video_files
		# self.max_sequence_len = max_sequence_len
		self.nb_classes = nb_classes
		self.cache_dir = cache_dir
		self.keyframe = keyframe

	def flow(self, batch_size):
		return VideoIterator(
			self, 
			self.video_files,
			# self.max_sequence_len,
			self.nb_classes,
			self.cache_dir,
			batch_size=batch_size,
			keyframe = self.keyframe)


########################################################################################################
def train_model(keyframe_status):
	"""Load data, compile, fit, evaluate model, and predict labels.
	Args:
		model: model name.
		y_categorical: whether to use the original label or one-hot label. True for classification models. False for regression models.
		max_len: the number of frames for each video.
		get_cross_validation: whether to cross validate. 
		non_zero: whether to use the non-zero data. If true 
	Returns:
		loss_mean: loss for this model.
		acc_mean: accuracy for classification model.
		classes: predications. Predication for all the videos is using cross validation.
		y_test: test ground truth. Equal to all labels if using cross validation."""

	n_classes = 101

	files = os.listdir(frames_dir)
	train_files = files[0:10000]
	val_files = files[10001:]

	model = raw_model(n_classes)

	# print train_files, len(train_files)
	datagenT = MyDataGenerator(video_files=train_files,
		nb_classes=n_classes,
		cache_dir=frames_dir,
		keyframe = keyframe_status)

	# # Then make a validation generator
	datagenV = MyDataGenerator(video_files=val_files,
		nb_classes=n_classes,
		cache_dir=frames_dir,
		keyframe = keyframe_status)

	nb_epoch = 50
	batch_size = 10
	samples_per_epoch = 10000/batch_size +1

	#checkpointer = ModelCheckpoint(out_model_file, monitor='loss', verbose=0, save_best_only=True, mode='auto')
	# checkpointer = ModelCheckpoint(out_model_file, monitor='acc', verbose=0, save_best_only=True, mode='auto')
	# model.fit_generator(datagenT.flow(batch_size=batch_size), 
	# 	nb_epoch=nb_epoch, 
	# 	samples_per_epoch=samples_per_epoch, 
	# 	verbose=1,
	# 	validation_data=datagenV.flow(batch_size=batch_size),
	# 	validation_steps=3000/batch_size +1)
	model.fit_generator(datagenT.flow(batch_size=batch_size), 
		nb_epoch=nb_epoch, 
		samples_per_epoch=samples_per_epoch, 
		verbose=1)
	loss, acc = model.evaluate_generator(datagenV.flow(batch_size=10), 3000/batch_size +1)
	print loss, acc
	return loss, acc

if __name__ == '__main__':
	train_model(True)
