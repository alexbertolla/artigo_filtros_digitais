import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import color
from matplotlib import pyplot as plt

qtd_quadrante = 16
caminho_imagem_original = '../banco_imagens/hz_1235134-PPT.jpg'
imagem_original = img_as_float(imread(caminho_imagem_original, as_gray=True))
linha, coluna = imagem_original.shape

print(linha/(qtd_quadrante/2))


print('FIM TESTE DIVIDIR IMAGEM')