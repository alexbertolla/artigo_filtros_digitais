import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import color
from matplotlib import pyplot as plt

lagarta = 'imagem_original.jpg'
lena = 'imagem_lena.png'

imagem_rgb = imread(lagarta)
imagem_rgb = img_as_float(imagem_rgb)

linha, coluna, _ = imagem_rgb.shape

quadrante1 = imagem_rgb[:int(linha/2), :int(coluna/2)]
quadrante2 = imagem_rgb[:int(linha/2), int(coluna/2):]
quadrante3 = imagem_rgb[int(linha/2):, :int(coluna/2)]
quadrante4 = imagem_rgb[int(linha/2):, int(coluna/2):]

imsave('quadrante_1.jpg', img_as_ubyte(quadrante1))
imsave('quadrante_2.jpg', img_as_ubyte(quadrante2))
imsave('quadrante_3.jpg', img_as_ubyte(quadrante3))
imsave('quadrante_4.jpg', img_as_ubyte(quadrante4))

plt.title('Divisão da Imagem')
plt.subplot(2, 2, 1)
plt.imshow(quadrante1)
plt.subplot(2, 2, 2)
plt.imshow(quadrante2)
plt.subplot(2, 2, 3)
plt.imshow(quadrante3)
plt.subplot(2, 2, 4)
plt.imshow(quadrante4)
plt.show()
#print(quadrante1)



print('FIM DIVISÃO IMAGEM')