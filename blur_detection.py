import numpy as np 
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import cv2
import sys

def detect_blur_fft(image, size=60, thresh=15,vis=False):
    (h,w) = image.shape
    #get center coordinate
    (cX, cY) = (int(w/2.0), int(h/2.0))
    #transform to frequency domain
    fft = np.fft.fft2(image)
    #shift zero-frequency (DC) component to center: = avg value of pixel frequencies, i.e. zero frequency
    #makes analysis easier- low frequency info closer to center, higher further out...
    #vertical lines -> one dot in horizantal direction (horizantal freq)
    #horizantal ines -> one dow in vertical direction
    fftShift = np.fft.fftshift(fft)
    if vis==True:
        # compute the magnitude spectrum of the 
        # transform
        magnitude = 20 * np.log(np.abs(fftShift))
        # display the original input image
        (fig, ax) = plt.subplots(1, 2, )
        ax[0].imshow(np.abs(image), cmap="gray")
        ax[0].set_title("Original")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        # display the magnitude image
        ax[1].imshow(np.abs(fftShift), cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        # show our plots
        
    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    #shift inverse fourier back to top left corner to transform back
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)
    # compute the magnitude spectrum of the reconstructed image,
    # then compute the mean of the magnitude values
    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    # the image will be considered "blurry" if the mean value of the
    # magnitudes is less than the threshold value
    print("mean: "+ str(mean))
    return (mean, mean <= thresh)


def variance_of_laplacian(image, thresh = 100.0):
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return (fm <= thresh)