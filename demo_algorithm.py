import numpy as np 
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
from compPhaseBatch import compPhaseBatch
from skimage import color
# import pdb
import re

def key_frame(data_dir, mFrame, top_k):
	reScale=[120,160]
	imgDir = os.listdir(data_dir)
	# imgDir = sorted(img_list, key=lambda k: map(int, os.path.splitext(k)[0]), reverse=True)
	# imgDir = imgDir[3:]

	nFrame = len(imgDir)
	# print imgDir[:5]
	                       

	batch = np.zeros((reScale[0],reScale[1],mFrame))
	for i in range(mFrame):
		# img = cv2.imread(imgDir[i])
		# print imgDir[i]
		img = cv2.imread(os.path.join(data_dir, imgDir[i]))
		img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
		batch[:,:,i] = cv2.resize(img,(reScale[1],reScale[0]))
	print 'number of videos: ',nFrame
	mmnorm = []
	for i in range(nFrame-mFrame):
		j = i+mFrame
		curFrame = cv2.imread(os.path.join(data_dir, imgDir[i]))

		batch[:,:,0:mFrame-2]=batch[:,:,1:mFrame-1]
		img = cv2.cvtColor(curFrame, cv2.COLOR_RGB2GRAY )
		batch[:,:,mFrame-1] = cv2.resize(img,(reScale[1],reScale[0]))

		motionMap=compPhaseBatch(batch)
		m_norm = np.linalg.norm(motionMap)
		mmnorm.append(m_norm)

		# plt.figure(1)
		# plt.imshow(curFrame)
		# plt.show(block=False)
		# plt.pause(.001)

		# plt.figure(2)
		# plt.imshow(motionMap, cmap='gray')
		# plt.show(block=False)
		# plt.pause(.001)
	# x=range(len(mmnorm))
	# plt.scatter(x,mmnorm)
	# plt.show()

	top_idx = sorted(range(len(mmnorm)), key=lambda k: mmnorm[k], reverse=True)
	sorted_top_idx = sorted(top_idx[:top_k])
	select_frames = []
	for i in range(top_k):
		select_frames.append(imgDir[sorted_top_idx[i]+mFrame/2])
	return select_frames

# dataPath='d/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/frames'
# key_frames_dir = 'd/home-4/ytian27@jhu.edu/scratch/yetian/C3D-TCN-Keras/key_frames'
# # dataPath = 'train-ids_8/4-2FNtn7m28[ 37 621  47 145  23  29 101]'
# mFrame=5
# top_k = 16
# folder_list = [f for f in os.listdir(dataPath)]
# n=0
# for video_id, videdo_dir in enumerate(folder_list):
# 	n +=1
# 	if os.path.isfile(os.path.join(key_frames_dir, videdo_dir))==False:
# 		os.mkdir(os.path.join(key_frames_dir, videdo_dir))

# 	select_frames = key_frame(data_dir, mFrame, top_k)





	