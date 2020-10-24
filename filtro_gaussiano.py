from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
import os
from skimage.filters import gaussian

dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'

dir_imagens_ruido_gaussiano_filtro_gaussiano = 'imagens_ruido_gaussiano_filtro_gaussiano'
dir_imagens_ruido_spekle_filtro_gaussiano = 'imagens_ruido_spekle_filtro_gaussiano'
dir_imagens_ruido_sal_e_pimenta_filtro_gaussiano = 'imagens_ruido_sal_e_pimenta_filtro_gaussiano'

lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_ruido_gaussiano)
for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_ruido_gaussiano = imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True)
    imagem_ruido_spekle = imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True)
    imagem_ruido_sal_e_pimenta = imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True)

    sigma = 3
    imagem_ruido_gaussiano_filtrada = gaussian(imagem_ruido_gaussiano, sigma=sigma, multichannel=False, mode='constant', cval=0)
    imagem_ruido_spekle_filtrada = gaussian(imagem_ruido_spekle, sigma=sigma, multichannel=False, mode='constant', cval=0)
    imagem_ruido_sal_e_pimenta_filtrada = gaussian(imagem_ruido_sal_e_pimenta, sigma=sigma, multichannel=False, mode='constant', cval=0)

    imagem_ruido_gaussiano_filtrada = img_as_ubyte(imagem_ruido_gaussiano_filtrada)
    imagem_ruido_spekle_filtrada = img_as_ubyte(imagem_ruido_spekle_filtrada)
    imagem_ruido_sal_e_pimenta_filtrada = img_as_ubyte(imagem_ruido_sal_e_pimenta_filtrada)

    imsave(dir_imagens_ruido_gaussiano_filtro_gaussiano +'/'+ nome_imagem, imagem_ruido_gaussiano_filtrada)
    imsave(dir_imagens_ruido_spekle_filtro_gaussiano + '/' + nome_imagem, imagem_ruido_spekle_filtrada)
    imsave(dir_imagens_ruido_sal_e_pimenta_filtro_gaussiano + '/' + nome_imagem, imagem_ruido_sal_e_pimenta_filtrada)

    print(nome_imagem)

print('FIM FILTRO GAUSSIANO')