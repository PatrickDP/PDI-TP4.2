# LETRA C: FAZ A ROTULAGEM DE COMPONENTES CONECTADOS

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# ALGORITMO DE ROTULAGEM DE COMPONENTES CONECTADOS - MÉTODO DE BUSCA POR LARGURA
# - ENTRADA:    IMAGEM LIMIARIZADA;
# - SAIDA:      IMAGEM ROTULADA;

# VERIFICA SE A VIZINHANÇA ESTÁ NO LIMITE
def is_bound_pixel(img, i, j):
    if (i < 0) or (j < 0):
        return False
    if (i > img.shape[0]-1) or (j > img.shape[1]-1):
        return False
    
    return True

# ENCONTRA OS POSSIVEIS VIZINHOS DO PIXEL ANALISADO EM QUESTÃO
def find_neighbors(img):
    final_pixel_neighbor = np.zeros((img.shape), dtype=object)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            p_up = (i-1, j)
            p_down = (i+1, j)
            p_left = (i, j-1)
            p_right = (i, j+1)
            pixel_neighbor = []

            if is_bound_pixel(img, p_up[0], p_up[1]):
                pixel_neighbor.append((p_up))
            if is_bound_pixel(img, p_down[0], p_down[1]):
                pixel_neighbor.append((p_down))
            if is_bound_pixel(img, p_left[0], p_left[1]):
                pixel_neighbor.append((p_left))
            if is_bound_pixel(img, p_right[0], p_right[1]):
                pixel_neighbor.append((p_right))
                
            final_pixel_neighbor[i, j] = pixel_neighbor
    
    return final_pixel_neighbor
    
# FAZ O PROCESSO DE ROTULAGEM
def labeling_breadthSearch(thresh_img):
    label_img = thresh_img.copy()
    label = 0
    queue = []
    pixel_neighbor = find_neighbors(label_img)
    
    for i in range(label_img.shape[0]):
        for j in range(label_img.shape[1]):
                            
            p = (i, j)
            if(label_img[p] == 255):
                label += 1
                label_img[p] = label
                queue.append((p))
                
                while queue:
                    q = queue.pop()
                    for q in pixel_neighbor[q]:
                        if(label_img[q] == 255):
                            label_img[q] = label
                            queue.append(q)
                            
    # print(label_img)
    plt.title('label_img'), plt.axis('off'), plt.imshow(label_img, cmap='gray'), plt.show()
    
    return label_img
    
    
