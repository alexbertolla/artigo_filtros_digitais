from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import numpy as np
from matplotlib import pylab
import os
import shutil

def add_ruido_gaussiano(imagem_original, sigma):
    #sigma = 0.05
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano


dir_imagens_redimensionadas = 'imagens_originais'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'

shutil.rmtree(dir_imagens_ruido_gaussiano, ignore_errors=True)
os.mkdir(dir_imagens_ruido_gaussiano)

lista_imagens_originais = os.listdir(dir_imagens_redimensionadas)
for nome_imagem in lista_imagens_originais:
    imagem_original = imread(dir_imagens_redimensionadas +'/'+ nome_imagem, as_gray=True)

    linha, coluna = imagem_original.shape

    imagem_ruidosa = np.copy(imagem_original)
    imagem_ruidosa[:int(linha / 2), :int(coluna / 2)] = add_ruido_gaussiano(imagem_original[:int(linha/2), :int(coluna/2)], 0.05)
    imagem_ruidosa[:int(linha / 2), int(coluna / 2):] = add_ruido_gaussiano(imagem_original[:int(linha/2), int(coluna/2):], 0.10)
    imagem_ruidosa[int(linha / 2):, :int(coluna / 2)] = add_ruido_gaussiano(imagem_original[int(linha/2):, :int(coluna/2)], 0.15)
    imagem_ruidosa[int(linha / 2):, int(coluna / 2):] = add_ruido_gaussiano(imagem_original[int(linha/2):, int(coluna/2):], 0.00)
    imagem_ruidosa = img_as_ubyte(imagem_ruidosa)
    imsave(dir_imagens_ruido_gaussiano + '/' + nome_imagem, imagem_ruidosa)

    print(dir_imagens_ruido_gaussiano +'/'+ nome_imagem)


print('FIM ADD RUIDO GAUSSIANO')