import numpy as np
import random
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import os


def add_ruido_sal_e_pimenta(imagem_original, prob):
    #prob = 0.05
    imagem_ruidosa = np.zeros(imagem_original.shape)
    thres = 1 - prob

    for i in range(imagem_original.shape[0]):
        for j in range(imagem_original.shape[1]):
            rdn = random.random()
            if rdn < prob:
                imagem_ruidosa[i][j] = 0
            elif rdn > thres:
                imagem_ruidosa[i][j] = 1
            else:
                imagem_ruidosa[i][j] = imagem_original[i][j]
    return imagem_ruidosa


dir_imagens_originais = 'imagens_originais'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'

lista_imagens_originais = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens_originais:
    imagem_original = imread(dir_imagens_originais +'/'+ nome_imagem, as_gray=True)
    imagem_ruidosa = np.copy(imagem_original)

    linha, coluna = imagem_original.shape

    imagem_ruidosa[:int(linha / 2), :int(coluna / 2)] = add_ruido_sal_e_pimenta(imagem_original[:int(linha / 2), :int(coluna / 2)], 0.05)
    imagem_ruidosa[:int(linha / 2), int(coluna / 2):] = add_ruido_sal_e_pimenta(imagem_original[:int(linha / 2), int(coluna / 2):], 0.10)
    imagem_ruidosa[int(linha / 2):, :int(coluna / 2)] = add_ruido_sal_e_pimenta(imagem_original[int(linha / 2):, :int(coluna / 2)], 0.15)
    imagem_ruidosa = img_as_ubyte(imagem_ruidosa)
    imsave(dir_imagens_ruido_sal_e_pimenta +'/'+ nome_imagem, imagem_ruidosa)
    print(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem)


print('FIM ADD RUIDO SAL E PIMENTA')