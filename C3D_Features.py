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
from demo_algorithm import key_frame
dim_ordering = K.image_dim_ordering()
print "[Info] image_dim_order (from default ~/.keras/keras.json)={}".format(
		dim_ordering)
backend = dim_ordering


data_dir = '/home/ye/Works/C3D-TCN-Keras/frames'
feature_dir = '/home/ye/Works/C3D-TCN-Keras/features/key-frame-c3d'

# data_dir = '/data/xiang/C3D-TCN-Keras/frames'
# feature_dir = '/data/xiang/C3D-TCN-Keras/features/C3D'

mFrame=5
top_k = 16

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

	if backend == 'th':
		model_weight_filename = os.path.join(model_dir, 'sports1M_weights_th.h5')
		model_json_filename = os.path.join(model_dir, 'sports1M_weights_th.json')
	else:
		model_weight_filename = os.path.join(model_dir, 'sports1M_weights_tf.h5')
		model_json_filename = os.path.join(model_dir, 'sports1M_weights_tf.json')

	print("[Info] Reading model architecture...")
	model = model_from_json(open(model_json_filename, 'r').read())
	#model = c3d_model.get_model(backend=backend)

	# visualize model
	model_img_filename = os.path.join(model_dir, 'c3d_model.png')
	if not os.path.exists(model_img_filename):
		from keras.utils import plot_model
		plot_model(model, to_file=model_img_filename)

	print("[Info] Loading model weights...")
	model.load_weights(model_weight_filename)
	print("[Info] Loading model weights -- DONE!")
	model.compile(loss='mean_squared_error', optimizer='sgd')

	print("[Info] Loading labels...")
	with open('sports1m/labels.txt', 'r') as f:
		labels = [line.strip() for line in f.readlines()]
	print('Total labels: {}'.format(len(labels)))

	print("[Info] Loading a sample video...")

	fold_names = [f for f in os.listdir(data_dir)]
	n = 0
	for video_id, video_dir in enumerate(fold_names):
		n += 1
		# print os.path.join(feature_dir,video_dir[2:])
		# print os.path.isfile(os.path.join(feature_dir,video_dir[2:]))
		feature_name = video_dir[2:]+'.npy'
		if os.path.isfile(os.path.join(feature_dir,feature_name)):
			print n,'/',len(fold_names),' ', video_dir, ' existed'
		else:
			start = time.time()
			video_path = os.path.join(data_dir, video_dir)

			sort_img = key_frame(video_path, mFrame, top_k)  # key frame

			# sort_img = np.sort(os.listdir(video_path))
			# print len(sort_img)

		#     if len(sort_img)<min_len:
		#     	min_len = len(sort_img)
		#     	min_id = video_idfea
		#     	min_dir = video_dir
		# print min_len, min_id, min_dir
		# shutil.rmtree(os.path.join(data_dir,min_dir))

			video_img = []
			for imgs in sort_img:
				img = cv2.imread(os.path.join(video_path,imgs))
				# print img.shape
				video_img.append(cv2.resize(img, (171, 128)))
			video_img = np.array(video_img, dtype=np.float32)
			# print video_img
			video_img -= np.array([ 89.62182617,97.80722809,101.37097168])
			# print video_img
			# print 'video_id','video_img', video_img.shape
			X = np.zeros((1,16,112,112,3))


			# for i in range((len(sort_img)/8)-1):
				# img16 = np.expand_dims(video_img[i*8:(i+2)*8,8:120,30:142,:], axis=0)
				# print 'img16', img16.shape
				# X = np.concatenate((X, img16),axis=0)
			# print 'X', X.shape

			img16 = np.expand_dims(video_img[0:16,8:120,30:142,:], axis=0)
			X = img16
			print 'X', X.shape

			int_model1 = c3d_model.get_int_model(model=model, layer='fc6', backend=backend)
			# int_model2 = c3d_model.get_int_model(model=model, layer='fc7', backend=backend)
			# int_model3 = c3d_model.get_int_model(model=model, layer='pool5', backend=backend)
			# int_model4 = c3d_model.get_int_model(model=model, layer='conv5b', backend=backend)
			# int_model5 = c3d_model.get_int_model(model=model, layer='conv5a', backend=backend)
			# int_model6 = c3d_model.get_int_model(model=model, layer='pool4', backend=backend)
			# int_model7 = c3d_model.get_int_model(model=model, layer='conv4b', backend=backend)
			# int_model8 = c3d_model.get_int_model(model=model, layer='conv4a', backend=backend)
			# int_model9 = c3d_model.get_int_model(model=model, layer='conv3b', backend=backend)
			# int_model10 = c3d_model.get_int_model(model=model, layer='conv3a', backend=backend)

			int_output1 = int_model1.predict_on_batch(X)
			# int_output2 = int_model2.predict_on_batch(X)
			# int_output3 = int_model3.predict_on_batch(X)
			# int_output4 = int_model4.predict_on_batch(X)
			# int_output5 = int_model5.predict_on_batch(X)
			# int_output6 = int_model6.predict_on_batch(X)
			# int_output7 = int_model7.predict_on_batch(X)
			# int_output8 = int_model8.predict_on_batch(X)
			# int_output9 = int_model9.predict_on_batch(X)
			# int_output10 = int_model10.predict_on_batch(X)
			# int_output = int_output[0, ...]
			end = time.time()
			video_time = end - start
			print n,'/',len(fold_names),' ', video_dir,': ', int_output1.shape, ' | ',' time: ', video_time,'s'
			np.save(os.path.join(feature_dir,video_dir[2:]),int_output1)

			# print 'fc6', int_output1.shape
			# print 'fc7', int_output2.shape
			# print 'pool5', int_output3.shape
			# print 'conv5b', int_output4.shape
			# print 'conv5a', int_output5.shape
			# print 'pool4', int_output6.shape
			# print 'conv4b', int_output7.shape
			# print 'conv4a', int_output8.shape
			# print 'conv3b', int_output9.shape
			# print 'conv3a', int_output10.shape


if __name__ == '__main__':
	main()