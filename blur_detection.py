import numpy as np 
import matplotlib.pyplot as plt
from imutils import paths
import argparse
import cv2
import sys
import pywt
import os

#try HAAR wavelet?

def detect_blur_fft(image, imgname, size=60, thresh=0,vis=True):
    (h,w) = image.shape
    #get center coordinate
    (cX, cY) = (int(w/2.0), int(h/2.0))
    #transform to frequency domain
    fft = np.fft.fft2(image)
    #shift zero-frequency (DC) component to center: = avg value of pixel frequencies, i.e. zero frequency
    #makes analysis easier- low frequency info closer to center, higher further out...
    #vertical lines -> one dot in horizantal direction (horizantal freq)
    #horizantal ines -> one dow in vertical direction3
    fftShift = np.fft.fftshift(fft)
    if vis:
        magnitude = 20 * np.log(np.abs(fftShift))
        output_folder = 'fft_results'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Display the original input image and magnitude spectrum
        (fig, ax) = plt.subplots(1, 2)

        ax[0].imshow(np.abs(image), cmap="gray")
        ax[0].set_title("Original")
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].imshow(magnitude, cmap="gray")
        ax[1].set_title("Magnitude Spectrum")
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        # Save the figure to the specified folder
        base_filename = os.path.basename(imgname)[:-4]  # Remove the file extension
        plot_filename = f"{base_filename}_spectrum.png" 
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close(fig)
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
    # Using cv2.putText() method 
     # Save the figure
    return (mean, mean <= thresh)


def variance_of_laplacian(image, thresh = 10.0):
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    #laplacian = cv2.convertScaleAbs(laplacian)
    cv2.imwrite('laplacian_image.jpg', laplacian)
    print("fm: "+str(fm))
    return (fm <= thresh)


def hwd_blur_detect(img, threshold = 35, minZero = 0.0005):
    
    # Convert image to grayscale
    
    M, N = img.shape
    
    # Crop input image to be 3 divisible by 2
    img = img[0:int(M/16)*16, 0:int(N/16)*16]
    
    # Step 1, compute Haar wavelet of input image
    LL1,(LH1,HL1,HH1)= pywt.dwt2(img, 'haar')
    # Another application of 2D haar to LL1
    LL2,(LH2,HL2,HH2)= pywt.dwt2(LL1, 'haar') 
    # Another application of 2D haar to LL2
    LL3,(LH3,HL3,HH3)= pywt.dwt2(LL2, 'haar')
    
    # Construct the edge map in each scale Step 2
    E1 = np.sqrt(np.power(LH1, 2)+np.power(HL1, 2)+np.power(HH1, 2))
    E2 = np.sqrt(np.power(LH2, 2)+np.power(HL2, 2)+np.power(HH2, 2))
    E3 = np.sqrt(np.power(LH3, 2)+np.power(HL3, 2)+np.power(HH3, 2))
    
    M1, N1 = E1.shape

    # Sliding window size level 1
    sizeM1 = 8
    sizeN1 = 8
    
    # Sliding windows size level 2
    sizeM2 = int(sizeM1/2)
    sizeN2 = int(sizeN1/2)
    
    # Sliding windows size level 3
    sizeM3 = int(sizeM2/2)
    sizeN3 = int(sizeN2/2)
    
    # Number of edge maps, related to sliding windows size
    N_iter = int((M1/sizeM1)*(N1/sizeN1))
    
    Emax1 = np.zeros((N_iter))
    Emax2 = np.zeros((N_iter))
    Emax3 = np.zeros((N_iter))
    
    
    count = 0
    
    # Sliding windows index of level 1
    x1 = 0
    y1 = 0
    # Sliding windows index of level 2
    x2 = 0
    y2 = 0
    # Sliding windows index of level 3
    x3 = 0
    y3 = 0
    
    # Sliding windows limit on horizontal dimension
    Y_limit = N1-sizeN1
    
    while count < N_iter:
        # Get the maximum value of slicing windows over edge maps 
        # in each level
        Emax1[count] = np.max(E1[x1:x1+sizeM1,y1:y1+sizeN1])
        Emax2[count] = np.max(E2[x2:x2+sizeM2,y2:y2+sizeN2])
        Emax3[count] = np.max(E3[x3:x3+sizeM3,y3:y3+sizeN3])
        
        # if sliding windows ends horizontal direction
        # move along vertical direction and resets horizontal
        # direction
        if y1 == Y_limit:
            x1 = x1 + sizeM1
            y1 = 0
            
            x2 = x2 + sizeM2
            y2 = 0
            
            x3 = x3 + sizeM3
            y3 = 0
            
            count += 1
        
        # windows moves along horizontal dimension
        else:
                
            y1 = y1 + sizeN1
            y2 = y2 + sizeN2
            y3 = y3 + sizeN3
            count += 1
    
    # Step 3
    EdgePoint1 = Emax1 > threshold
    EdgePoint2 = Emax2 > threshold
    EdgePoint3 = Emax3 > threshold
    
    # Rule 1 Edge Pojnts
    EdgePoint = EdgePoint1 + EdgePoint2 + EdgePoint3
    
    n_edges = EdgePoint.shape[0]
    
    # Rule 2 Dirak-Structure or Astep-Structure
    DAstructure = (Emax1[EdgePoint] > Emax2[EdgePoint]) * (Emax2[EdgePoint] > Emax3[EdgePoint]);
    
    # Rule 3 Roof-Structure or Gstep-Structure
    
    RGstructure = np.zeros((n_edges))

    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax1[i] < Emax2[i] and Emax2[i] < Emax3[i]:
                RGstructure[i] = 1
                
    # Rule 4 Roof-Structure
    
    RSstructure = np.zeros((n_edges))

    for i in range(n_edges):
        if EdgePoint[i] == 1:
            if Emax2[i] > Emax1[i] and Emax2[i] > Emax3[i]:
                RSstructure[i] = 1

    # Rule 5 Edge more likely to be in a blurred image 

    BlurC = np.zeros((n_edges))

    for i in range(n_edges):
        if RGstructure[i] == 1 or RSstructure[i] == 1:
            if Emax1[i] < threshold:  
                BlurC[i] = 1                        
        
    # Step 6
    Per = np.sum(DAstructure)/np.sum(EdgePoint)
    
    # Step 7
    if (np.sum(RGstructure) + np.sum(RSstructure)) == 0:
        BlurExtent = 100
    else:
        BlurExtent = np.sum(BlurC) / (np.sum(RGstructure) + np.sum(RSstructure))
    
    return Per, (Per < minZero)