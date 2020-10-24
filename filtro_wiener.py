from skimage import color, restoration
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import os
import numpy as np
from matplotlib import pylab as plt

dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'

dir_imagens_ruido_gaussiano_filtro_wiener = 'imagens_ruido_gaussiano_filtro_wiener'
dir_imagens_ruido_spekle_filtro_wiener = 'imagens_ruido_spekle_filtro_wiener'
dir_imagens_ruido_sal_e_pimenta_filtro_wiener = 'imagens_ruido_sal_e_pimenta_filtro_wiener'

lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_ruido_gaussiano)
for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_ruido_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle = img_as_float(imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True))

    n = 3
    psf = np.ones((n, n)) / n**2

    imagem_ruido_gaussiano_filtrada, _ = restoration.unsupervised_wiener(imagem_ruido_gaussiano, psf)
    imagem_ruido_spekle_filtrada, _ = restoration.unsupervised_wiener(imagem_ruido_spekle, psf)
    imagem_ruido_sal_e_pimenta_filtrada, _ = restoration.unsupervised_wiener(imagem_ruido_sal_e_pimenta, psf)

    imagem_ruido_gaussiano_filtrada = img_as_ubyte(imagem_ruido_gaussiano_filtrada)
    imagem_ruido_spekle_filtrada = img_as_ubyte(imagem_ruido_spekle_filtrada)
    imagem_ruido_sal_e_pimenta_filtrada = img_as_ubyte(imagem_ruido_sal_e_pimenta_filtrada)

    imsave(dir_imagens_ruido_gaussiano_filtro_wiener + '/' + nome_imagem, imagem_ruido_gaussiano_filtrada)
    imsave(dir_imagens_ruido_spekle_filtro_wiener + '/' + nome_imagem, imagem_ruido_spekle_filtrada)
    imsave(dir_imagens_ruido_sal_e_pimenta_filtro_wiener + '/' + nome_imagem, imagem_ruido_sal_e_pimenta_filtrada)
    print(nome_imagem)


print('FIM FILTRO WIENER')