import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import os

def add_ruido_spekle(imagem_original, sigma):
    imagem_ruidosa = np.copy(imagem_original)
    gauss = np.random.normal(0, sigma, imagem_original.size)

    if len(imagem_original.shape) == 3:
        gauss = gauss.reshape(imagem_original.shape[0], imagem_original.shape[1], imagem_original.shape[2]).astype('uint8')
    else:
        gauss = gauss.reshape(imagem_original.shape[0], imagem_original.shape[1]).astype('uint8')
    imagem_ruidosa = img_as_ubyte(imagem_original) + img_as_ubyte(imagem_original) * gauss
    imagem_ruidosa = img_as_float(imagem_ruidosa)
    return imagem_ruidosa


dir_imagens_originais = 'imagens_originais'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'

lista_imagens_originais = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens_originais:
    imagem_original = imread(dir_imagens_originais +'/'+ nome_imagem, as_gray=True)
    imagem_ruidosa = np.copy(imagem_original)

    linha, coluna = imagem_original.shape

    imagem_ruidosa[:int(linha / 2), :int(coluna / 2)] = add_ruido_spekle(imagem_original[:int(linha / 2), :int(coluna / 2)], 0.7)
    imagem_ruidosa[:int(linha / 2), int(coluna / 2):] = add_ruido_spekle(imagem_original[:int(linha / 2), int(coluna / 2):], 1.0)
    imagem_ruidosa[int(linha / 2):, :int(coluna / 2)] = add_ruido_spekle(imagem_original[int(linha / 2):, :int(coluna / 2)], 1.3)
    imagem_ruidosa = img_as_ubyte(imagem_ruidosa)
    imsave(dir_imagens_ruido_spekle +'/'+ nome_imagem, imagem_ruidosa)
    print(dir_imagens_ruido_spekle + '/' + nome_imagem)


print('FIM ADD RUIDO SPEKLE')