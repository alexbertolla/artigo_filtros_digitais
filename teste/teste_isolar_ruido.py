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

imagem_ruidosa = add_ruido_gaussiano(imagem_original, 1.0)

imagem_ruido_isolado = imagem_ruidosa - imagem_original
# imagem_ruido_isolado = img_as_ubyte(imagem_ruido_isolado)


#imagem_original = np.zeros([728, 728], dtype=np.float)
#imagem_ruidosa = add_ruido_gaussiano(imagem_original, 1.0)
#imagem_ruido_isolado = imagem_ruidosa - imagem_original

print(imagem_original.dtype)
print(imagem_ruidosa.dtype)
print(imagem_ruido_isolado.dtype)
print(imagem_ruido_isolado)


pylab.figure()
pylab.subplot(1, 5, 1)
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(1, 5, 2)
pylab.imshow(imagem_ruidosa, cmap='gray')

pylab.subplot(1, 5, 3)
pylab.imshow(imagem_ruido_isolado, cmap='gray')

pylab.subplot(1, 5, 4)
pylab.hist(imagem_ruido_isolado.flat)

pylab.subplot(1, 5, 5)
pylab.hist(img_as_ubyte(imagem_ruido_isolado.flat))

pylab.show()
print('FIM')