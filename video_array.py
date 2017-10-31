#!/usr/bin/env python

import matplotlib
matplotlib.use('Agg')
# from keras.models import model_from_json
import os
import cv2
# import skvideo.io
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import shutil
# import keras.backend as K

import subprocess


# data_dir = '/home/ye/Works/C3D-TCN-Keras/frames'
# feature_dir = '/media/ye/youtube-8/UCF101-npy'

data_dir = '/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/frames'
feature_dir = '/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/frames-npy'

def main():


	fold_names = [f for f in os.listdir(data_dir)]
	n = 0
	for video_id, video_dir in enumerate(fold_names):
		print video_dir
		n += 1
		feature_name = video_dir[2:]+'.npy'
		if os.path.isfile(os.path.join(feature_dir,feature_name)):
			print n,'/',len(fold_names),' ', video_dir, ' existed'
		else:
			# try:
			start = time.time()
			video_path = os.path.join(data_dir, video_dir)
			sort_img = np.sort(os.listdir(video_path))

			video_img = []
			for imgs in sort_img:
				img = cv2.imread(os.path.join(video_path,imgs))
				# print 'image shape',img.shape
				video_img.append(cv2.resize(img, (112, 112)))
			video_img = np.array(video_img, dtype=np.float32)

			# video_img -= np.array([ 89.62182617,97.80722809,101.37097168])
			# print 'video_img', video_img.shape

			end = time.time()
			video_time = end - start
			print n,'/',len(fold_names),' ',video_dir, ': ', video_img.shape, ' | ',' time: ', video_time,'s'
			np.save(os.path.join(feature_dir,video_dir[2:]),video_img)
				# continue
			# except KeyError:
			# 	print '------', video_dir[2:], 'error', '--------'


if __name__ == '__main__':
	main()