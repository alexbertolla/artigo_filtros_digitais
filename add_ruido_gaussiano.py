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
    print('Add ruído sigma = ', sigma)
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano


dir_imagens_redimensionadas = 'banco_imagens'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'

shutil.rmtree(dir_imagens_ruido_gaussiano, ignore_errors=True)
os.mkdir(dir_imagens_ruido_gaussiano)

lista_ruido = [0.00, 0.05, 0.10, 0.15]

lista_imagens_originais = os.listdir(dir_imagens_redimensionadas)
for nome_imagem in lista_imagens_originais:
    imagem_original = img_as_float(imread(dir_imagens_redimensionadas +'/'+ nome_imagem, as_gray=True))

    linha, coluna = imagem_original.shape

    imagem_ruidosa = np.zeros(imagem_original.shape)

    imagem_dividida = dividir_imagem(imagem_original)


    subq1 = dividir_imagem(imagem_dividida[0])
    subq1[0] = add_ruido_gaussiano(subq1[0], 0.0)
    subq1[1] = add_ruido_gaussiano(subq1[1], 0.05)
    subq1[2] = add_ruido_gaussiano(subq1[2], 0.10)
    subq1[3] = add_ruido_gaussiano(subq1[3], 0.15)

    subq2 = dividir_imagem(imagem_dividida[1])
    subq2[0] = add_ruido_gaussiano(subq2[0], 0.20)
    subq2[1] = add_ruido_gaussiano(subq2[1], 0.25)
    subq2[2] = add_ruido_gaussiano(subq2[2], 0.30)
    subq2[3] = add_ruido_gaussiano(subq2[3], 0.35)

    subq3 = dividir_imagem(imagem_dividida[2])
    subq3[0] = add_ruido_gaussiano(subq3[0], 0.40)
    subq3[1] = add_ruido_gaussiano(subq3[1], 0.45)
    subq3[2] = add_ruido_gaussiano(subq3[2], 0.50)
    subq3[3] = add_ruido_gaussiano(subq3[3], 0.55)

    subq4 = dividir_imagem(imagem_dividida[3])
    subq4[0] = add_ruido_gaussiano(subq4[0], 0.60)
    subq4[1] = add_ruido_gaussiano(subq4[1], 0.65)
    subq4[2] = add_ruido_gaussiano(subq4[2], 0.70)
    subq4[3] = add_ruido_gaussiano(subq4[3], 0.75)

    q1 = np.zeros(imagem_dividida[0].shape)
    q1 = montar_imagem(subq1, q1)

    q2 = np.zeros(imagem_dividida[1].shape)
    q2 = montar_imagem(subq2, q2)

    q3 = np.zeros(imagem_dividida[2].shape)
    q3 = montar_imagem(subq3, q3)

    q4 = np.zeros(imagem_dividida[3].shape)
    q4 = montar_imagem(subq4, q4)

    #imagem_ruidosa = montar_imagem([q1, q2, q3, q4], imagem_ruidosa)
    imagem_ruidosa[:int(linha / 2), :int(coluna / 2)] = add_ruido_gaussiano(imagem_original[:int(linha / 2), :int(coluna / 2)], 0.00)  # QUADRANTE 1
    imagem_ruidosa[:int(linha / 2), int(coluna / 2):] = add_ruido_gaussiano(imagem_original[:int(linha / 2), int(coluna / 2):], 0.05)  # QUADRANTE 2
    imagem_ruidosa[int(linha / 2):, int(coluna / 2):] = add_ruido_gaussiano(imagem_original[int(linha / 2):, int(coluna / 2):], 0.10)  # QUADRANTE 3
    imagem_ruidosa[int(linha / 2):, :int(coluna / 2)] = add_ruido_gaussiano(imagem_original[int(linha / 2):, :int(coluna / 2)], 0.15)  # QUADRANTE 4


    imagem_ruidosa = img_as_ubyte(imagem_ruidosa)
    imsave(dir_imagens_ruido_gaussiano + '/' + nome_imagem, imagem_ruidosa)

    print(dir_imagens_ruido_gaussiano +'/'+ nome_imagem)


print('FIM ADD RUIDO GAUSSIANO')