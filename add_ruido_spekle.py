import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import color
from matplotlib import pyplot as plt

lagarta = 'imagem_original.jpg'
lena = 'imagem_lena.png'

imagem_rgb = imread(lagarta)
imagem_rgb = img_as_float(imagem_rgb)

imagem_ruidosa = np.copy(imagem_rgb)
gauss = np.random.normal(0, 0.7, imagem_rgb.size)
gauss = gauss.reshape(imagem_rgb.shape[0], imagem_rgb.shape[1], imagem_rgb.shape[2]).astype('uint8')
#.astype('uint8')
imagem_ruidosa = img_as_ubyte(imagem_rgb) + img_as_ubyte(imagem_rgb) * gauss


print(gauss)
print(imagem_ruidosa)

plt.subplot(2, 2, 1)
plt.imshow(imagem_rgb)
plt.subplot(2, 2, 2)
plt.imshow(imagem_ruidosa)
plt.show()

print('FIM ADD RUIDO SPEKLE')