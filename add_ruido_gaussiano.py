from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import numpy as np
from matplotlib import pylab
import os
import shutil
import random

def dividir_imagem(imagem):
    array_imagem = []
    l, c = imagem.shape
    array_imagem.append(imagem[:int(l / 2), :int(c / 2)])  # QUADRANTE 1
    array_imagem.append(imagem[:int(l / 2), int(c / 2):])  # QUADRANTE 2
    array_imagem.append(imagem[int(l / 2):, int(c / 2):])  # QUADRANTE 3
    array_imagem.append(imagem[int(l / 2):, :int(c / 2)])  # QUADRANTE 4

    return array_imagem

def montar_imagem(array_imagem, imagem_montada):
    l, c = imagem_montada.shape
    imagem_montada[:int(l / 2), :int(c / 2)] = array_imagem[0]  # QUADRANTE 1
    imagem_montada[:int(l / 2), int(c / 2):] = array_imagem[1]  # QUADRANTE 2
    imagem_montada[int(l / 2):, int(c / 2):] = array_imagem[3]  # QUADRANTE 3
    imagem_montada[int(l / 2):, :int(c / 2)] = array_imagem[2]  # QUADRANTE 4

    return imagem_montada

def add_ruido_gaussiano(imagem_original, sigma):
    #sigma = 0.05
    #print('Add ruído sigma = ', sigma)
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano


dir_imagens_redimensionadas = 'banco_imagens'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'

shutil.rmtree(dir_imagens_ruido_gaussiano, ignore_errors=True)
os.mkdir(dir_imagens_ruido_gaussiano)

lista_ruido = [0.00, 0.05, 0.10, 0.15]

lista_imagens_originais = os.listdir(dir_imagens_redimensionadas)
total_imagem = len(lista_imagens_originais)
aux_total_imagem = 1
for nome_imagem in lista_imagens_originais:
    print(str(aux_total_imagem) + ' de ' + str(total_imagem) + ' imagens.')
    aux_total_imagem += 1
    
    imagem_original = img_as_float(imread(dir_imagens_redimensionadas +'/'+ nome_imagem, as_gray=True))

    linha, coluna = imagem_original.shape

    imagem_ruidosa = np.zeros(imagem_original.shape)

    imagem_dividida = dividir_imagem(imagem_original)

    #imagem_ruidosa = montar_imagem([q1, q2, q3, q4], imagem_ruidosa)
    imagem_ruidosa[:int(linha / 2), :int(coluna / 2)] = add_ruido_gaussiano(imagem_original[:int(linha / 2), :int(coluna / 2)], 0.00)  # QUADRANTE 1
    imagem_ruidosa[:int(linha / 2), int(coluna / 2):] = add_ruido_gaussiano(imagem_original[:int(linha / 2), int(coluna / 2):], 0.05)  # QUADRANTE 2
    imagem_ruidosa[int(linha / 2):, int(coluna / 2):] = add_ruido_gaussiano(imagem_original[int(linha / 2):, int(coluna / 2):], 0.10)  # QUADRANTE 3
    imagem_ruidosa[int(linha / 2):, :int(coluna / 2)] = add_ruido_gaussiano(imagem_original[int(linha / 2):, :int(coluna / 2)], 0.15)  # QUADRANTE 4


    imagem_ruidosa = img_as_ubyte(imagem_ruidosa)
    imsave(dir_imagens_ruido_gaussiano + '/' + nome_imagem, imagem_ruidosa)

    #print(dir_imagens_ruido_gaussiano +'/'+ nome_imagem)


print('FIM ADD RUIDO GAUSSIANO')