from cv2 import threshold
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import letterB as b

# CASO 1, 2, 3 
img = cv.imread('test_images/' + 'bolhas.png')[:,:,::-1]
img2 = cv.imread('test_images/' + 'coins-01.jpg')[:,:,::-1]
img3 = cv.imread('test_images/' + 'rice.png')[:,:,::-1]

# LETRA A: REMOVENDO RUÍDO DA IMAGEM PELO FILTRO GAUSSIANO
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
gaussianBlur_img = cv.GaussianBlur(gray_img, [5, 5], 0)

gray_img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
gaussianBlur_img2 = cv.GaussianBlur(gray_img2, [7, 7], 0)

gray_img3 = cv.cvtColor(img3, cv.COLOR_RGB2GRAY)
gaussianBlur_img3 = cv.GaussianBlur(gray_img3, [5, 5], 0)

plt.subplot(3, 3, 1)
plt.title('img')
plt.axis('OFF')
plt.imshow(img)

plt.subplot(3, 3, 2)
plt.title('gray_img')
plt.axis('OFF')
plt.imshow(gray_img, cmap='gray')

plt.subplot(3, 3, 3)
plt.title('gaussianBlur_img')
plt.axis('OFF')
plt.imshow(gaussianBlur_img, cmap='gray')

plt.subplot(3, 3, 4)
plt.title('img2')
plt.axis('OFF')
plt.imshow(img2)

plt.subplot(3, 3, 5)
plt.title('gray_img2')
plt.axis('OFF')
plt.imshow(gray_img2, cmap='gray')

plt.subplot(3, 3, 6)
plt.title('gaussianBlur_img2')
plt.axis('OFF')
plt.imshow(gaussianBlur_img2, cmap='gray')

plt.subplot(3, 3, 7)
plt.title('img3')
plt.axis('OFF')
plt.imshow(img3)

plt.subplot(3, 3, 8)
plt.title('gray_img3')
plt.axis('OFF')
plt.imshow(gray_img3, cmap='gray')

plt.subplot(3, 3, 9)
plt.title('gaussianBlur_img3')
plt.axis('OFF')
plt.imshow(gaussianBlur_img3, cmap='gray')

plt.show()

# RESPONSÁVEL EM REMOVER O FUNDO DA IMAGEM.
b.remove_background(img, gaussianBlur_img, False)
b.remove_background(img2, gaussianBlur_img2, True)
b.remove_background(img3, gaussianBlur_img3, False)