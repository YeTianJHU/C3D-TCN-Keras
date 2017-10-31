import numpy as np 
import cv2
from skimage import color
from scipy import signal 

def compPhaseBatch(input_batch):
	n_img = input_batch.shape[2]
	motionmap = np.zeros((input_batch.shape[0],input_batch.shape[1]))

	for i in range(n_img-1):

		curFrame1 = input_batch[:,:,i]
		curFrame2 = input_batch[:,:,i+1]
		myFFT1 = np.fft.fft2(curFrame1)
		Amp1 = abs(myFFT1)
		Phase1 = np.angle(myFFT1,deg=False)
		# print Phase1
		myFFT2 = np.fft.fft2(curFrame2)
		Amp2 = abs(myFFT2)
		Phase2 = np.angle(myFFT2,deg=False)
		mMap1 = abs(np.fft.ifft2(np.multiply((Amp1-Amp2),np.exp(1j*Phase1))))
		mMap2 = abs(np.fft.ifft2(np.multiply((Amp1-Amp2),np.exp(1j*Phase2))))
		mMap = np.multiply(mMap1,mMap2)

		motionmap = motionmap + mMap

	motionmap = signal.convolve2d(motionmap, generate_gauss2d((15,15),3.0), mode="same")
	m_motionmap = np.zeros(motionmap.shape, np.double)
	cv2.normalize(motionmap, m_motionmap, 1.0, 0.0, cv2.NORM_MINMAX)

	return m_motionmap


def generate_gauss2d(shape=(3, 3), sigma=0.5, order=0):
	"""
	This function will return 2D Gaussian filter for convolution
	Args:
		shape (tuple): The shape of Gaussian filter, which is (3, 3) by default.
		sigma (scalar): A scalar which represents both x and y expectation.
		order (scalar): {0, 1, 2} The order of the filter. An order of 0 corresponds to 
						convolution with a Gaussian kernel. An order of 1 or 1 corresponds 
						to convolution with the first or second derivatives of a Gaussian.
						By default, it will return the first derivative.
	Returns:
		gauss2D (numpy.ndarray): Matrix represents the output filter. Order==0 will output
								 only one filter; order==1 will output 2 filters w.r.t. 
								 x-axis and y-axis.
	"""

	if order not in (0, 1):
		raise "The order of Gaussian filter can only be 0, 1.\n"
	if order == 0:
		m, n = [(ss - 1.) / 2. for ss in shape]
		y,x = np.ogrid[-m:m+1,-n:n+1]
		gauss2D = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
		gauss2D[ gauss2D < np.finfo(gauss2D.dtype).eps*gauss2D.max() ] = 0
		sumh = gauss2D.sum()
		if sumh != 0:
			gauss2D /= sumh
		return gauss2D
	else:
		r_y = (shape[0] - 1.) / 2.
		r_x = (shape[1] - 1.) / 2.
		y, x = np.ogrid[-r_y:r_y+1, -r_x:r_x+1]
		d_x = -x / (2 * math.pi * sigma**2) * np.exp(-(x*x + y*y) / (2 * sigma))
		d_y = -y / (2 * math.pi * sigma**2) * np.exp(-(x*x + y*y) / (2 * sigma))
		return [d_x, d_y]