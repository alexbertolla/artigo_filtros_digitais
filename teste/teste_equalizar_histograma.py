import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import exposure
from matplotlib import pylab as pylab

caminho_imagem = '../banco_imagens/hz_1235134-PPT.jpg'
imagem = img_as_float(imread(caminho_imagem, as_gray=True))
imagem_equalizada = exposure.equalize_adapthist(imagem)


imagem = img_as_ubyte(imagem)
imagem_equalizada = img_as_ubyte(imagem_equalizada)

pylab.figure()
pylab.subplot(2, 2, 1)
pylab.axis('off')
pylab.imshow(imagem, cmap='gray')

pylab.subplot(2, 2, 2)
pylab.axis('on')
pylab.hist(imagem.flat, bins=256, range=(0, 255), color='black')

pylab.subplot(2, 2, 3)
pylab.axis('off')
pylab.imshow(imagem_equalizada, cmap='gray')

pylab.subplot(2, 2, 4)
pylab.axis('on')
pylab.hist(imagem_equalizada.flat, bins=256, range=(0, 255), color='black')

pylab.show()