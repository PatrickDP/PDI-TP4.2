import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# LETRA A: REMOVENDO RUÍDO DA IMAGEM PELO FILTRO GAUSSIANO
img = cv.imread('test_images/' + 'coins-01.jpg')[:,:,::-1]
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gaussianBlur_img = cv.GaussianBlur(gray_img, [7, 7], 0)

plt.subplot(1, 3, 1)
plt.title('img')
plt.axis('OFF')
plt.imshow(img)

plt.subplot(1, 3, 2)
plt.title('gray_img')
plt.axis('OFF')
plt.imshow(gray_img, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('gaussianBlur_img')
plt.axis('OFF')
plt.imshow(gaussianBlur_img, cmap='gray')

plt.show()
