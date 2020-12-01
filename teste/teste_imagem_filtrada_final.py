import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy import fftpack
from matplotlib import pylab


imagem_passa_alta = img_as_float(imread('../imagens_filtro_passa_alta/0.1/hz_1235134-PPT.jpg', as_gray=True))
imagem_passa_baixa = img_as_float(imread('../imagens_filtro_passa_baixa/0.1/hz_1235134-PPT.jpg', as_gray=True))

imagem_final = imagem_passa_alta + imagem_passa_baixa

pylab.figure()
pylab.imshow(imagem_final, cmap='gray')
pylab.show()

#linha, coluna = imagem_passa_alta.shape

print('FIM TESTE IMAGEM FILTRADA FINAL')