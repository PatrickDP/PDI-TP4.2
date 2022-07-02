import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def generate_core_map(img):
    # cv.imshow('img', img)
    # cv.waitKey(0)

    maximum_value = np.max(img)
    hue = 0
    range_hue = 0
    
    spaceColor_value = []
    arr_color = []
    
    hsv_img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    
    for i in range(maximum_value):
        if(hue <= 180):
            hue += range_hue
            spaceColor_value= (hue, 255, 255)
            range_hue = int(np.floor(180 / maximum_value))
            arr_color.append(spaceColor_value)
            
    for i in range(hsv_img.shape[0]):
        for j in range(hsv_img.shape[1]):
            p = (i, j)
            
            if(hsv_img[i, j, 2] > 0):
                hsv_img[i, j] = [int(0 * 180), int(255 * 255), int(255 * 255)]
                print(hsv_img[p])
    cv.imshow('img', hsv_img)
    cv.waitKey(0)