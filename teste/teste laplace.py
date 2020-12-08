import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from matplotlib import pylab
import os
from scipy import fftpack
from skimage.filters import laplace
import cv2 as cv

def gerar_spectro(imagem):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

caminho_imagem_original = '../banco_imagens/hz_1234055-PPT.jpg'
imagem_original = cv.imread(caminho_imagem_original)
imagem_original = cv.cvtColor(imagem_original, cv.COLOR_BGR2GRAY)
spectro_original = gerar_spectro(imagem_original)


#linha, coluna = imagem_original.shape

imagem_laplace = cv.Laplacian(imagem_original, cv.CV_8U, ksize=5)
spectro_laplace = gerar_spectro(imagem_laplace)



pylab.figure()
pylab.subplot(3, 4, 1)
pylab.axis('off')
pylab.title('Imagem Original')
pylab.imshow(img_as_ubyte(imagem_original), cmap='gray')

pylab.subplot(3, 4, 2)
pylab.axis('off')
pylab.title('Spectro Original')
pylab.imshow(spectro_original, cmap='gray')

pylab.subplot(3, 4, 3)
pylab.axis('off')
pylab.title('Imagem Laplace')
pylab.imshow(imagem_laplace, cmap='gray')

pylab.subplot(3, 4, 4)
pylab.axis('off')
pylab.title('Spectro Laplace')
pylab.imshow(spectro_laplace, cmap='gray')


pylab.show()