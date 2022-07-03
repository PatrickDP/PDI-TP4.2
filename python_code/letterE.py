# LETRA E: MOSTRA O HISTOGRAMA DA IMAGEM ROTULADA E A AREA DE CADA PIXELS

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def generate_histogram(img):
    img_ravel = np.ravel(img)
    ir = np.where(img_ravel > 0)
    
    histogram = plt.hist(img_ravel[ir], 256, [0, 256])
    
    arr_pixelsObjects = histogram[0]
    ap = np.where(arr_pixelsObjects > 0)
    
    plt.title('Histogram'), plt.show()
    print("Area of objects in number of pixels:"), print(arr_pixelsObjects[ap])