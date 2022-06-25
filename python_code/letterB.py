# RESOLUÇÃO DA LETRA B

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def generate_histogram(img):
    histogram = cv.calcHist([img], [0], None, [256], [0, 256])
    
    plt.plot(histogram)
    plt.xlim([0, 256])
    plt.show()
    
def remove_background(img, gaussianBlur_img, what_img):
    mask_name = 0
    mask_img = []
    final_img = []
    
    # PARA O CASO DA PRIMEIRA IMAGEM (JÁ LIMIARIZADA E PRECISA APENAS DO MÉTODO CLOSING)
    if(what_img == 1):
        kernel = cv.getStructuringElement(cv.MORPH_RECT, [5, 1])
        erode_img = cv.erode(img, kernel, iterations=1)
        
        kernel_2 = cv.getStructuringElement(cv.MORPH_CROSS, [25, 25])
        closing_Eimg = cv.morphologyEx(erode_img, cv.MORPH_CLOSE, kernel_2)
        
        kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, [7, 7])
        erode_CEimg = cv.erode(closing_Eimg, kernel_3, iterations=1)
 
        bitwise_img = cv.bitwise_or(img, erode_CEimg, mask=None)
        
        kernel_4 = cv.getStructuringElement(cv.MORPH_RECT, [1, 8])
        
        mask_name = "bolhas_mask.png"
        final_img = cv.morphologyEx(bitwise_img, cv.MORPH_CLOSE, kernel_4)
        mask_img = final_img.copy()
                
        plt.figure('bolhas.png')
        
        plt.subplot(2, 3, 1), plt.title('img'), plt.axis('OFF'), plt.plot(), plt.imshow(img, cmap='gray')
        plt.subplot(2, 3, 2), plt.title('erode_img'), plt.axis('OFF'), plt.plot(), plt.imshow(erode_img, cmap='gray')
        plt.subplot(2, 3, 3), plt.title('closing_Eimg'), plt.axis('OFF'), plt.plot(), plt.imshow(closing_Eimg, cmap='gray')
        
        plt.subplot(2, 3, 4), plt.title('erode_CEimg'), plt.axis('OFF'), plt.plot(), plt.imshow(erode_CEimg, cmap='gray')
        plt.subplot(2, 3, 5), plt.title('bitwise_img'), plt.axis('OFF'), plt.plot(), plt.imshow(bitwise_img, cmap='gray')
        plt.subplot(2, 3, 6), plt.title('final_image'), plt.axis('OFF'), plt.plot(), plt.imshow(final_img, cmap='gray')
        
        plt.show()      
    
    # ... SEGUNDA IMAGEM (USO DO CANNY PARA DETECTAR AS BORDAS DA MOEDA BRANCA, OP. MORFOLÓGICOS E BITWISE)
    elif(what_img == 2):
        canny_img = cv.Canny(gaussianBlur_img, 100, 0)
        threshold, thresh_img = cv.threshold(gaussianBlur_img, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY_INV)
        bitwise_img = cv.bitwise_xor(canny_img, thresh_img, mask=None)
        
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, [19, 19])
        closing_BCimg = cv.morphologyEx(bitwise_img, cv.MORPH_CLOSE, kernel)
        
        kernel_2 = cv.getStructuringElement(cv.MORPH_ELLIPSE, [75, 75])
        opening_Cimg = cv.morphologyEx(closing_BCimg, cv.MORPH_OPEN, kernel_2)
        
        bitwise_TOimg = cv.bitwise_xor(thresh_img, opening_Cimg, mask=None)
        bitwise_OBimg = cv.bitwise_or(opening_Cimg, bitwise_TOimg, mask=None)
    
        mask_name = "coins-01_mask.jpg" 
        mask_img = bitwise_OBimg.copy()
        final_img = cv.bitwise_and(img, img, mask=bitwise_OBimg)
        
        plt.figure('coins-01.jpg')
        
        plt.subplot(3, 3, 1), plt.title('gaussianBlur_img'), plt.axis('OFF'), plt.plot(), plt.imshow(gaussianBlur_img, cmap='gray')
        plt.subplot(3, 3, 2), plt.title('canny_img'), plt.axis('OFF'), plt.plot(), plt.imshow(canny_img, cmap='gray')
        plt.subplot(3, 3, 3), plt.title('thresh_img'), plt.axis('OFF'), plt.plot(), plt.imshow(thresh_img, cmap='gray')
        
        plt.subplot(3, 3, 4), plt.title('bitwise_img'), plt.axis('OFF'), plt.plot(), plt.imshow(bitwise_img, cmap='gray')
        plt.subplot(3, 3, 5), plt.title('closing_img'), plt.axis('OFF'), plt.plot(), plt.imshow(closing_BCimg, cmap='gray')
        plt.subplot(3, 3, 6), plt.title('opening_img'), plt.axis('OFF'), plt.plot(), plt.imshow(opening_Cimg, cmap='gray')
        
        plt.subplot(3, 3, 7), plt.title('bitwise_TOimg'), plt.axis('OFF'), plt.plot(), plt.imshow(bitwise_TOimg, cmap='gray')
        plt.subplot(3, 3, 8), plt.title('bitwise_OBimg'), plt.axis('OFF'), plt.plot(), plt.imshow(bitwise_OBimg, cmap='gray')
        plt.subplot(3, 3, 9), plt.title('final_img'), plt.axis('OFF'), plt.plot(), plt.imshow(final_img, cmap='gray')
        
        plt.show()
    # ... TERCEIRA IMAGEM (USANDO LIMIAR ADAPTATIVO E O MÉTODO OPENING)
    elif(what_img == 3):
        thresh_img = cv.adaptiveThreshold(gaussianBlur_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 199, 3)
        
        kernel = cv.getStructuringElement(cv.MORPH_CROSS, [3, 3])
        opening_img = cv.morphologyEx(thresh_img, cv.MORPH_OPEN, kernel)
        
        mask_name = "rice_mask.png"
        mask_img = opening_img.copy()
        final_img = cv.bitwise_and(img, img, mask=opening_img)
        
        plt.figure('rice.png')
        
        plt.subplot(2, 2, 1), plt.title('gaussianBlur_img'), plt.axis('OFF'), plt.plot(), plt.imshow(gaussianBlur_img, cmap='gray')
        plt.subplot(2, 2, 2), plt.title('thresh_img'), plt.axis('OFF'), plt.plot(), plt.imshow(thresh_img, cmap='gray')
        
        plt.subplot(2, 2, 3), plt.title('opening_img'), plt.axis('OFF'), plt.plot(), plt.imshow(opening_img, cmap='gray')
        plt.subplot(2, 2, 4), plt.title('final_img'), plt.axis('OFF'), plt.plot(), plt.imshow(final_img, cmap='gray')
        
        plt.show()
    else:
        print('ERROR - RETURNING FUNCTION...')
        return
    
    cv.imwrite('output_images/%s' %(mask_name), mask_img)
    return final_img