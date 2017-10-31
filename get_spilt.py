import numpy as np 
import os
from shutil import copyfile

# file_dir = '/home/ye/Works/C3D-TCN-Keras/features/C3D'
# txt_dir = '/home/ye/Works/C3D-TCN-Keras/ucfTrainTestlist'
# spilt_dir = '/home/ye/Works/C3D-TCN-Keras/features/C3D-split'
file_dir = '/home/ye/Works/C3D-TCN-Keras/features/VGG'
txt_dir = '/home/ye/Works/C3D-TCN-Keras/ucfTrainTestlist'
# spilt_dir = '/home/ye/Works/C3D-TCN-Keras/features/VGG-split'
spilt_dir = '/media/ye/youtube-8/UCF101-VGG-SPLIT'

# txt_name = ['testlist01.txt','testlist02.txt','testlist03.txt','trainlist01.txt','trainlist02.txt','trainlist03.txt']
txt_name = ['testlist01.txt','testlist02.txt']



for n in range(len(txt_name)):
	print 
	print txt_name[n]
	print 

	txt_path = os.path.join(txt_dir, txt_name[n])
	video_list = np.genfromtxt(txt_path, delimiter=' ', dtype=None, usecols=(0))

	fold_name = os.path.splitext(os.path.basename(txt_name[n]))[0]
	fold_path = os.path.join(spilt_dir, fold_name)
	if (os.path.isdir(fold_path) == False):
		os.mkdir(fold_path)

	for ids, row in enumerate(video_list):
		file = os.path.splitext(os.path.basename(row))[0]+'.npy'
		if os.path.isfile(os.path.join(file_dir, file[2:])):
			copyfile(os.path.join(file_dir, file[2:]), os.path.join(fold_path, file[2:]))
			print ids, '/', len(video_list), '|', file[2:], ' copied'
		else:
			print ids, '/', len(video_list), '|', file[2:], '--------- error --------'