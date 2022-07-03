# LETRA B: REMOVE O FUNDO DA IMAGEM

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
    
def remove_background(img,  what_img):
    # CASO PARA A 2A IMAGEM
    if(what_img == 2):
        canny_img = cv.Canny(img, 100, 0)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [7, 7])
        canny_img = cv.morphologyEx(canny_img, cv.MORPH_CLOSE, kernel, iterations=2)
        
        img = canny_img.copy()
    # CASO PARA A 3A IMAGEM
    if(what_img == 3):
        canny_img = cv.Canny(img, 100, 113)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [3, 3])
        canny_img = cv.morphologyEx(canny_img, cv.MORPH_CLOSE, kernel, iterations=1)
        
        img = canny_img.copy()
    
    img_floodfill = img.copy()
    height, width = img_floodfill.shape[:2]
    mask = np.zeros((height+2, width+2), np.uint8)
    
    cv.floodFill(img_floodfill, mask, (0, 0), 255)
    
    inverse_floodFill = cv.bitwise_not(img_floodfill)    
    
    # APRIMORAMENTOS
    if(what_img == 1):
        foreground_img = cv.bitwise_or(img, inverse_floodFill)
    if(what_img == 2):
        foreground_img = cv.bitwise_or(img, inverse_floodFill)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [75, 75])
        foreground_img = cv.morphologyEx(foreground_img, cv.MORPH_OPEN, kernel, iterations=1)
        
        kernel2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, [3, 3])
        foreground_img = cv.erode(foreground_img, kernel2, iterations=2)
        foreground_img = cv.morphologyEx(foreground_img, cv.MORPH_OPEN, kernel)
    if(what_img == 3):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [3, 3])
        inverse_floodFill = cv.morphologyEx(inverse_floodFill, cv.MORPH_OPEN, kernel, iterations=2)
        inverse_floodFill = cv.dilate(inverse_floodFill, kernel, iterations=1)
       
        foreground_img = inverse_floodFill.copy()
    
    
    plt.subplot(2, 2, 1), plt.title('1'), plt.axis('OFF'), plt.imshow(img, cmap='gray')
    plt.subplot(2, 2, 2), plt.title('2'), plt.axis('OFF'), plt.imshow(img_floodfill, cmap='gray')
    plt.subplot(2, 2, 3), plt.title('3'), plt.axis('OFF'), plt.imshow(inverse_floodFill, cmap='gray')
    plt.subplot(2, 2, 4), plt.title('4'), plt.axis('OFF'), plt.imshow(foreground_img, cmap='gray')

    plt.show()

    return foreground_img