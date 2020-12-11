from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import numpy as np
from matplotlib import pylab
import random

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano

divisao = 2
dir_imagens = '../banco_imagens/hz_1235134-PPT.jpg'
imagem_original = img_as_float(imread(dir_imagens, as_gray=True))
imagem_ruido = np.zeros(imagem_original.shape)
lista_ruido = [0.00, 0.05, 0.10, 0.15]

l, c = imagem_ruido.shape
imagem_ruido[:int(l/2), :int(c/2)] = add_ruido_gaussiano(imagem_original[:int(l/2), :int(c/2)], 0.00 )#QUADRANTE 1
imagem_ruido[:int(l/2), int(c/2):] = add_ruido_gaussiano(imagem_original[:int(l/2), int(c/2):], 0.05) #QUADRANTE 2
imagem_ruido[int(l/2):, int(c/2):] = add_ruido_gaussiano(imagem_original[int(l/2):, int(c/2):], 0.10) #QUADRANTE 3
imagem_ruido[int(l/2):, :int(c/2)] = add_ruido_gaussiano(imagem_original[int(l/2):, :int(c/2)], 0.15) #QUADRANTE 4



pylab.figure()
pylab.subplot(1, 2, 1)
pylab.axis('off')
pylab.imshow(img_as_ubyte(imagem_original), cmap='gray')

pylab.subplot(1, 2, 2)
pylab.axis('off')
pylab.imshow(img_as_ubyte(imagem_ruido), cmap='gray')



pylab.show()