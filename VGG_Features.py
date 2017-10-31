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
import time
import shutil
import keras.backend as K

import subprocess
from keras.models import Sequential, Model 
from keras.layers import Input, merge, Dense, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dropout
from keras.regularizers import l2
# from keras.regularizers import ActivityRegularizer
from keras.layers.convolutional import Convolution2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras import backend as K
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image

dim_ordering = K.image_dim_ordering()
print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
		dim_ordering)
backend = dim_ordering


data_dir = '/home/ye/Works/C3D-TCN-Keras/frames'
feature_dir = '/home/ye/Works/C3D-TCN-Keras/features/VGG'

# data_dir = '/data/xiang/C3D-TCN-Keras/frames'
# feature_dir = '/data/xiang/C3D-TCN-Keras/features/VGG'

def main():
	show_images = False
	diagnose_plots = False
	model_dir = './models'
	global backend

	# override backend if provided as an input arg
	if len(sys.argv) > 1:
		if 'tf' in sys.argv[1].lower():
			backend = 'tf'
		else:
			backend = 'th'
	print "[Info] Using backend={}".format(backend)

# Load in the pre-trained model
	base_model = VGG16(weights='imagenet')
	a = base_model.get_layer('fc2')
	print a
	model = Model(input=base_model.input, output=base_model.get_layer('fc2').output)  
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# visualize model
	model_img_filename = os.path.join(model_dir, 'c3d_model.png')
	if not os.path.exists(model_img_filename):
		from keras.utils import plot_model
		plot_model(model, to_file=model_img_filename)



	print("[Info] Loading labels...")
	with open('sports1m/labels.txt', 'r') as f:
		labels = [line.strip() for line in f.readlines()]
	print('Total labels: {}'.format(len(labels)))

	print("[Info] Loading a sample video...")

	fold_names = [f for f in os.listdir(data_dir)]
	n = 0
	for video_id, video_dir in enumerate(fold_names):
		n += 1
		feature_name = video_dir[2:]+'.npy'
		if os.path.isfile(os.path.join(feature_dir,feature_name)):
			print n,'/',len(fold_names),' ', video_dir, ' existed'
		else:
			start = time.time()
			video_path = os.path.join(data_dir, video_dir)
			sort_img = np.sort(os.listdir(video_path))

			video_img = []
			for imgs in sort_img:
				img = cv2.imread(os.path.join(video_path,imgs))
				# print 'image shape',img.shape
				video_img.append(cv2.resize(img, (224, 224)))
			video_img = np.array(video_img, dtype=np.float32)
			video_img -= np.array([ 89.62182617,97.80722809,101.37097168])
			# print 'video_img', video_img.shape

			
			X = model.predict(video_img)

			end = time.time()
			video_time = end - start
			print n,'/',len(fold_names),' ',video_dir, ': ', X.shape, ' | ',' time: ', video_time,'s'
			np.save(os.path.join(feature_dir,video_dir[2:]),X)



if __name__ == '__main__':
	main()