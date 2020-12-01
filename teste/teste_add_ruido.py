from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import numpy as np
from matplotlib import pylab
import random

def dividir_imagem(imagem):
    array_imagem = []
    l, c = imagem.shape
    array_imagem.append(imagem[:int(l/2), :int(c/2)])
    array_imagem.append(imagem[:int(l/2), int(c/2):])
    array_imagem.append(imagem[int(l/2):, :int(c/2)])
    array_imagem.append(imagem[int(l/2):, int(c/2):])
    return array_imagem

def montar_imagem(array_imagem, imagem_montada):
    l, c = imagem_montada.shape
    imagem_montada[:int(l/2), :int(c/2)] = array_imagem[0]
    imagem_montada[:int(l/2), int(c/2):] = array_imagem[1]
    imagem_montada[int(l/2):, :int(c/2)] = array_imagem[2]
    imagem_montada[int(l/2):, int(c/2):] = array_imagem[3]
    return imagem_montada

divisao = 2
dir_imagens = '../banco_imagens/hz_imagem_64.jpg'
imagem_original = imread(dir_imagens, as_gray=True)
lista_ruido = [0.00, 0.05, 0.10, 0.15]

imagem_dividida = dividir_imagem(imagem_original)
nova_imagem = np.zeros(imagem_original.shape)
nova_imagem = montar_imagem(imagem_dividida, nova_imagem)

q1 = dividir_imagem(imagem_dividida[0])
novo_q1 = np.zeros(imagem_dividida[0].shape)
novo_q1 = montar_imagem(q1, novo_q1)

pylab.figure()
pylab.subplot(3, 2, 1)
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(3, 2, 2)
pylab.imshow(nova_imagem, cmap='gray')

pylab.subplot(3, 2, 3)
pylab.imshow(imagem_dividida[0], cmap='gray')

pylab.subplot(3, 2, 4)
pylab.imshow(q1[0], cmap='gray')



pylab.show()