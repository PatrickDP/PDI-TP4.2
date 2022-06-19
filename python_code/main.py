import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import letterB as b

# CASO 1, 2, 3 
img = cv.imread('test_images/' + 'bolhas.png')[:,:,::-1]
img2 = cv.imread('test_images/' + 'coins-01.jpg')[:,:,::-1]
img3 = cv.imread('test_images/' + 'rice.png')[:,:,::-1]

# LETRA A: REMOVENDO RUÍDO DA IMAGEM PELO FILTRO GAUSSIANO
gray_img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
gaussianBlur_img = cv.GaussianBlur(gray_img, [7, 7], 0)

gray_img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
gaussianBlur_img2 = cv.GaussianBlur(gray_img2, [7, 7], 0)

gray_img3 = cv.cvtColor(img3, cv.COLOR_RGBA2GRAY)
gaussianBlur_img3 = cv.GaussianBlur(gray_img3, [3, 3], 0)

plt.figure('images')

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

cv.imwrite('test_images/gaussianBlur_bolhas.png', gaussianBlur_img)
cv.imwrite('test_images/gaussianBlur_coins-01.jpg', gaussianBlur_img2)
cv.imwrite('test_images/gaussianBlur_rice.png', gaussianBlur_img3)


# B: RESPONSÁVEL EM REMOVER O FUNDO DA IMAGEM.
noBackground_image = b.remove_background(img, False, 1)
noBackground_image2 = b.remove_background(img2, gaussianBlur_img2, 2)
noBackground_image3 = b.remove_background(img3, gaussianBlur_img3, 3)

noBackground_image = cv.cvtColor(noBackground_image, cv.COLOR_RGBA2BGRA)
noBackground_image2 = cv.cvtColor(noBackground_image2, cv.COLOR_RGB2BGR)
noBackground_image3 = cv.cvtColor(noBackground_image3, cv.COLOR_RGBA2BGRA)

cv.imwrite('output_images/bolhas_result.png', noBackground_image)
cv.imwrite('output_images/coins-01_result.jpg', noBackground_image2)
cv.imwrite('output_images/rice_result.png', noBackground_image3)

