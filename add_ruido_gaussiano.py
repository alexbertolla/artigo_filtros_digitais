from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import os


def add_ruido_gaussiano(imagem_original):
    sigma = 0.05
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano


dir_imagens_redimensionadas = 'imagens_originais'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'

lista_imagens_originais = os.listdir(dir_imagens_redimensionadas)
for nome_imagem in lista_imagens_originais:
    imagem_original = imread(dir_imagens_redimensionadas +'/'+ nome_imagem, as_gray=True)
    imagem_ruido_gaussiano = add_ruido_gaussiano(imagem_original)
    imagem_ruido_gaussiano = img_as_ubyte(imagem_ruido_gaussiano)
    imsave(dir_imagens_ruido_gaussiano +'/'+ nome_imagem, imagem_ruido_gaussiano)
    print(dir_imagens_ruido_gaussiano +'/'+ nome_imagem)


print('FIM ADD RUIDO GAUSSIANO')