import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import letterB as b
import letterC as c
import letterE as e

# CASO 1, 2, 3 
img = cv.imread('test_images/' + 'bolhas.png')[:,:,::-1]
img2 = cv.imread('test_images/' + 'coins-01.png')[:,:,::-1]
img3 = cv.imread('test_images/' + 'rice.png')[:,:,::-1]

# LETRA A: REMOVENDO RUÍDO DA IMAGEM PELO FILTRO GAUSSIANO
gray_img = cv.cvtColor(img, cv.COLOR_RGBA2GRAY)

gray_img2 = cv.cvtColor(img2, cv.COLOR_RGBA2GRAY)
gaussianBlur_img2 = cv.GaussianBlur(gray_img2, [7, 7], 0)

gray_img3 = cv.cvtColor(img3, cv.COLOR_RGBA2GRAY)
gaussianBlur_img3 = cv.GaussianBlur(gray_img3, [3, 3], 0)

plt.figure('images')

plt.subplot(3, 3, 1), plt.title('img'), plt.axis('OFF'), plt.imshow(img)
plt.subplot(3, 3, 2), plt.title('gray_img'), plt.axis('OFF'), plt.imshow(gray_img, cmap='gray')

plt.subplot(3, 3, 4), plt.title('img2'), plt.axis('OFF'), plt.imshow(img2)
plt.subplot(3, 3, 5), plt.title('gray_img2'), plt.axis('OFF'), plt.imshow(gray_img2, cmap='gray')
plt.subplot(3, 3, 6), plt.title('gaussianBlur_img2'), plt.axis('OFF'), plt.imshow(gaussianBlur_img2, cmap='gray')

plt.subplot(3, 3, 7), plt.title('img3'), plt.axis('OFF'), plt.imshow(img3)
plt.subplot(3, 3, 8), plt.title('gray_img3'), plt.axis('OFF'), plt.imshow(gray_img3, cmap='gray')
plt.subplot(3, 3, 9), plt.title('gaussianBlur_img3'), plt.axis('OFF'), plt.imshow(gaussianBlur_img3, cmap='gray')

plt.show()

cv.imwrite('test_images/gaussianBlur_coins-01.png', gaussianBlur_img2)
cv.imwrite('test_images/gaussianBlur_rice.png', gaussianBlur_img3)

# B: RESPONSÁVEL EM REMOVER O FUNDO DA IMAGEM.
mask_img = b.remove_background(gray_img, 1)
cv.imwrite('output_images/bolhas_result.png', mask_img)

mask_img2 = b.remove_background(gaussianBlur_img2, 2)
foreground_img2 = cv.bitwise_and(img2, img2, mask=mask_img2)
foreground_img2 = cv.cvtColor(foreground_img2, cv.COLOR_RGBA2BGRA)
cv.imwrite('output_images/coins-01_result.png', foreground_img2)
cv.imwrite('output_images/coins-01_mask.png', mask_img2)

mask_img3 = b.remove_background(gaussianBlur_img3, 3)
foreground_img3 = cv.bitwise_and(img3, img3, mask=mask_img3)
foreground_img3 = cv.cvtColor(foreground_img3, cv.COLOR_RGBA2BGRA)
cv.imwrite('output_images/rice_result.png', foreground_img3)
cv.imwrite('output_images/rice_mask.png', mask_img3)


# C: FAZ A ROTULAÇÃO DE OBJETOS CONECTADOS POR BUSCA E LARGURA
thresh_img =  cv.imread('output_images/bolhas_result.png')
thresh_img2 =  cv.imread('output_images/coins-01_mask.png')
thresh_img3 =  cv.imread('output_images/rice_mask.png')

thresh_img = cv.cvtColor(thresh_img, cv.COLOR_BGR2GRAY)
thresh_img2 = cv.cvtColor(thresh_img2, cv.COLOR_BGR2GRAY)
thresh_img3 = cv.cvtColor(thresh_img3, cv.COLOR_BGR2GRAY)

label_img = c.labeling_breadthSearch(thresh_img)
label_img2 = c.labeling_breadthSearch(thresh_img2)
label_img3 = c.labeling_breadthSearch(thresh_img3)

cv.imwrite('output_images/bolhas_labeling.png', label_img)
cv.imwrite('output_images/coins-01_labeling.png', label_img2)
cv.imwrite('output_images/rice_labeling.png', label_img3)

# E: UTILIZA O HISTOGRAMA DA IMAGEM ROTULADA E MOSTRA A AREA EM PIXEL DE CADA OBJ
print("In labsel_img:\n"), e.generate_histogram(label_img)
print("\nIn label_img2:\n"), e.generate_histogram(label_img2)
print("\nIn label_img3:\n"), e.generate_histogram(label_img3)