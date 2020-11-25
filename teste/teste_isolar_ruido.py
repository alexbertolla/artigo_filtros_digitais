from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import numpy as np
from matplotlib import pylab


def add_ruido_gaussiano(imagem_original, sigma):
    #sigma = 0.05
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano


imagem_original = img_as_float(imread('../banco_imagens/hz_1224099-PPT.jpg', as_gray=True))
print(imagem_original.dtype)
imagem_ruidosa = add_ruido_gaussiano(imagem_original, 1.0)
print(imagem_ruidosa.dtype)
imagem_ruido = abs(imagem_original - imagem_ruidosa)
imagem_ruido = img_as_ubyte(imagem_ruido)
print(imagem_ruido)

pylab.figure()
pylab.subplot(1, 3, 1)
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(1, 3, 2)
pylab.imshow(imagem_ruidosa, cmap='gray')

pylab.subplot(1, 3, 3)
pylab.imshow(imagem_ruido, cmap='gray')

pylab.show()
print('FIM')