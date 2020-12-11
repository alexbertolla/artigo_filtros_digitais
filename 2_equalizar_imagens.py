import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import exposure
import os
from skimage.color import rgb2gray
from matplotlib import pylab

dir_banco_imagens = './banco_imagens/'

lista_imagens_originais = os.listdir(dir_banco_imagens)
total_imagem = len(lista_imagens_originais)
aux_total_imagem = 1
for nome_imagem in lista_imagens_originais:
    print(str(aux_total_imagem) + ' de ' + str(total_imagem) + ' imagens.')
    aux_total_imagem += 1
    imagem_original = img_as_float(imread(dir_banco_imagens + '/' + nome_imagem))

    imagem_equalizada = exposure.rescale_intensity(imagem_original)
    imagem_equalizada = img_as_ubyte(imagem_equalizada)

    imsave(dir_banco_imagens + nome_imagem, imagem_equalizada)
    #pylab.imshow(imagem_equalizada, cmap='gray')
    #pylab.show()





print('FIM REDIMENSIONAR')