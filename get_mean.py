#!/usr/bin/env python


import os
import cv2
import skvideo.io
import numpy as np
import sys
import time




data_dir = '/home/ye/Works/C3D-TCN-Keras/frames'
feature_dir = '/home/ye/Works/C3D-TCN-Keras/features/VGG16'


def main():

	fold_names = [f for f in os.listdir(data_dir)]
	mean = []

	for video_id, video_dir in enumerate(fold_names):
		print video_id, video_dir
		start = time.time()
		video_path = os.path.join(data_dir, video_dir)
		sort_img = np.sort(os.listdir(video_path))

		video_img = []
		for imgs in sort_img:
			img = cv2.imread(os.path.join(video_path,imgs))
			# print 'image shape',img.shape
			video_img.append(cv2.resize(img, (224, 224)))
		video_img = np.array(video_img, dtype=np.float32)
		# print 'video_img', video_img.shape
		a = np.mean(video_img, axis=0)
		b = np.mean(a,axis=0)
		c = np.mean(b,axis=0)
		mean.append(c)

		
		# mean_show = np.array(mean, dtype=np.float32)
		# print mean_show.shape
	mean_show = np.array(mean, dtype=np.float32)
	print mean_show.shape
	GRB_result = np.mean(mean_show,axis=0)
	print GRB_result
	np.save('mean',mean_show)


			

			# end = time.time()
			# video_time = end - start
			# print video_dir, ': ', X.shape, ' | ',' time: ', video_time,'s'
			# np.save(os.path.join(feature_dir,video_dir[2:]),X)



if __name__ == '__main__':
	main()