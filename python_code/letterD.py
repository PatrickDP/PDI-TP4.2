# LETRA D: MOSTRA A QUANTIDADES DE OBJETOS E COLORE A IMAGEM

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def generate_core_map(img):
    maximum_value = np.max(img)
    hue_counter = 0
    arr_color = []

    print("Objects counted:", maximum_value)
    
    hsv2rgb_img = np.zeros((1, 1, 3), dtype=np.uint8)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    
    for i in range(maximum_value):
        
        if(hue_counter < 180):
            hsv2rgb_img[::] = (hue_counter, 255, 255)
            
            arr_color.append(cv.cvtColor(hsv2rgb_img, cv.COLOR_HSV2RGB))
            hue_counter += int(np.floor(180 / maximum_value))
               
    rgb_img = np.zeros(img.shape, dtype=np.uint8)
    
    for k in range(maximum_value):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                p = (i, j, 2)    
                
                if(img[p] == (k + 1)):
                    rgb_img[i, j] = arr_color[k]
                    
    plt.title('Color Image'), plt.axis('OFF'), plt.imshow(rgb_img)
    plt.show()
        
    return rgb_img