''' RESOLUÇÃO DA LETRA B '''

from contextlib import closing
from cv2 import bitwise_and, dilate, threshold
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def generate_histogram(img):
    histogram = cv.calcHist([img], [0], None, [256], [0, 256])
    
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()
    
def remove_background(img, gray_image, whiteBackground):
    # A IMAGEM POSSUI FUNDO BRANCO
    if(whiteBackground == True):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
        ret, thresh_image = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    # A IMAGEM POSSUI FUNDO PRETO
    else:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        thresh_image = cv.adaptiveThreshold(gray_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 3)
        
    closing_thresh = cv.morphologyEx(thresh_image, cv.MORPH_CLOSE, kernel)
    
    canny_image = cv.Canny(gray_image, 75, 0)
    closing_canny = cv.morphologyEx(canny_image, cv.MORPH_CLOSE, kernel)
    
    mask_image = cv.bitwise_or(closing_thresh, closing_canny, mask=None)
    foreground_image = cv.bitwise_and(img, img, mask=mask_image)
    
    plt.subplot(2, 2, 1)
    plt.title('thresh_image')
    plt.axis('OFF')
    plt.plot()
    plt.imshow(thresh_image, cmap='binary_r')
    
    plt.subplot(2, 2, 2)
    plt.title('closing_thresh')
    plt.axis('OFF')
    plt.plot()
    plt.imshow(closing_thresh, cmap='binary_r')
    
    plt.subplot(2, 2, 3)
    plt.title('canny_image')
    plt.axis('OFF')
    plt.plot()
    plt.imshow(canny_image, cmap='binary_r')
    
    plt.subplot(2, 2, 4)
    plt.title('closing_canny')
    plt.axis('OFF')
    plt.plot()
    plt.imshow(closing_canny, cmap='binary_r')
    
    plt.show()
    
    plt.subplot(1, 2, 1)
    plt.title('mask_image')
    plt.axis('OFF')
    plt.plot()
    plt.imshow(mask_image, cmap='binary_r')
    
    plt.subplot(1, 2, 2)
    plt.title('foreground_image')
    plt.axis('OFF')
    plt.plot()
    plt.imshow(foreground_image, cmap='binary_r')
    
    plt.show()
    
    return foreground_image