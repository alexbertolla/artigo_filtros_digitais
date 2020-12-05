import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from scipy import fftpack
import os
import shutil

def gerar_spectro(transformata_imagem):
    shift_frq = fftpack.fftshift(transformata_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

def aplicar_filtro_passa_alta(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    spectro_imagem_original = gerar_spectro(discrete_transform_imagem)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)

    shift_frq[half_w - (int(half_w * porcentagem_corte)):half_w + (int(half_w * porcentagem_corte)) + 1, half_h - (int(half_h * porcentagem_corte)):half_h + (int(half_h * porcentagem_corte)) + 1] = 0
    imagem_filtrada = np.clip(fp.ifft2(fftpack.ifftshift(shift_frq)).real, 0, 255)

    return imagem_filtrada


lista_corte = (0.05, 0.07, 0.10, 0.13, 0.15)
dir_imagens_ruidosas = './imagens_ruido_gaussiano/'
dir_imagens_filtro_passa_alta = './imagens_filtro_passa_alta/'

shutil.rmtree(dir_imagens_filtro_passa_alta, ignore_errors=True)
os.mkdir(dir_imagens_filtro_passa_alta)

lista_imagens = os.listdir(dir_imagens_ruidosas)
for porcentagem_corte in lista_corte:
    os.mkdir(dir_imagens_filtro_passa_alta + str(porcentagem_corte))
    print(dir_imagens_filtro_passa_alta + str(porcentagem_corte))
    for nome_imagem in lista_imagens:
        print(nome_imagem)
        imagem_ruidosa = img_as_float(imread(dir_imagens_ruidosas + nome_imagem, as_gray=True))
        linha, coluna = imagem_ruidosa.shape

        imagem_filtrada_passa_alta = img_as_float(np.zeros(imagem_ruidosa.shape))

        imagem_filtrada_passa_alta[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_alta(
            imagem_ruidosa[:int(linha / 2), :int(coluna / 2)], porcentagem_corte)

        imagem_filtrada_passa_alta[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_alta(
            imagem_ruidosa[:int(linha / 2), int(coluna / 2):], porcentagem_corte)

        imagem_filtrada_passa_alta[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_alta(
            imagem_ruidosa[int(linha / 2):, :int(coluna / 2)], porcentagem_corte)

        imagem_filtrada_passa_alta[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_alta(
            imagem_ruidosa[int(linha / 2):, int(coluna / 2):], porcentagem_corte)

        imagem_filtrada_passa_alta = img_as_ubyte(imagem_filtrada_passa_alta)
        imsave(dir_imagens_filtro_passa_alta + str(porcentagem_corte) + '/' + nome_imagem, imagem_filtrada_passa_alta)


print('FIM FILTRO PASSA ALTA')