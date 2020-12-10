import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import color
from matplotlib import pyplot as plt

caminho_imagem_original = '../banco_imagens/hz_1235134-PPT.jpg'

imagem_rgb = imread(caminho_imagem_original)
imagem_rgb = img_as_float(imagem_rgb)

linha, coluna = imagem_rgb.shape

quadrante1 = imagem_rgb[:int(linha/2), :int(coluna/2)]
quadrante2 = imagem_rgb[:int(linha/2), int(coluna/2):]
quadrante3 = imagem_rgb[int(linha/2):, int(coluna/2):]
quadrante4 = imagem_rgb[int(linha/2):, :int(coluna/2)]



#imsave('quadrante_1.jpg', img_as_ubyte(quadrante1))
#imsave('quadrante_2.jpg', img_as_ubyte(quadrante2))
#imsave('quadrante_3.jpg', img_as_ubyte(quadrante3))
#imsave('quadrante_4.jpg', img_as_ubyte(quadrante4))

plt.figure()

plt.subplot(2, 5, 1)
plt.axis('off')
plt.title('Imagem Original')
plt.imshow(imagem_rgb, cmap='gray')

plt.subplot(2, 5, 2)
plt.axis('off')
plt.title('quadrante1')
plt.imshow(quadrante1, cmap='gray')

plt.subplot(2, 5, 3)
plt.axis('off')
plt.title('quadrante2')
plt.imshow(quadrante2, cmap='gray')

plt.subplot(2, 5, 4)
plt.axis('off')
plt.title('quadrante3')
plt.imshow(quadrante3, cmap='gray')

plt.subplot(2, 5, 5)
plt.axis('off')
plt.title('quadrante4')
plt.imshow(quadrante4, cmap='gray')

plt.show()
#print(quadrante1)



print('FIM DIVISÃO IMAGEM')