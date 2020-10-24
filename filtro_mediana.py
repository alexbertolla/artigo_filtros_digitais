from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import os
from skimage.filters import median
from skimage.morphology import disk

dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'

dir_imagens_ruido_gaussiano_filtro_mediana = 'imagens_ruido_gaussiano_filtro_mediana'
dir_imagens_ruido_spekle_filtro_mediana = 'imagens_ruido_spekle_filtro_mediana'
dir_imagens_ruido_sal_e_pimenta_filtro_mediana = 'imagens_ruido_sal_e_pimenta_filtro_mediana'

lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_ruido_gaussiano)
for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_ruido_gaussiano = imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True)
    imagem_ruido_spekle = imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True)
    imagem_ruido_sal_e_pimenta = imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True)

    janela = 3
    imagem_ruido_gaussiano_filtrada = median(imagem_ruido_gaussiano, disk(janela), mode='constant', cval=0.0)
    imagem_ruido_spekle_filtrada = median(imagem_ruido_spekle, disk(janela), mode='constant', cval=0.0)
    imagem_ruido_sal_e_pimenta_filtrada = median(imagem_ruido_sal_e_pimenta, disk(janela), mode='constant', cval=0.0)

    imagem_ruido_gaussiano_filtrada = img_as_ubyte(imagem_ruido_gaussiano_filtrada)
    imagem_ruido_spekle_filtrada = img_as_ubyte(imagem_ruido_spekle_filtrada)
    imagem_ruido_sal_e_pimenta_filtrada = img_as_ubyte(imagem_ruido_sal_e_pimenta_filtrada)

    imsave(dir_imagens_ruido_gaussiano_filtro_mediana +'/'+ nome_imagem, imagem_ruido_gaussiano_filtrada)
    imsave(dir_imagens_ruido_spekle_filtro_mediana + '/' + nome_imagem, imagem_ruido_spekle_filtrada)
    imsave(dir_imagens_ruido_sal_e_pimenta_filtro_mediana + '/' + nome_imagem, imagem_ruido_sal_e_pimenta_filtrada)

    print(nome_imagem)

print('FIM FILTRO MEDIANA')

