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

def aplicar_filtro_passa_baixa(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    spectro_imagem_original = gerar_spectro(discrete_transform_imagem)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    shift_frq_low = np.copy(shift_frq)

    shift_frq_low[half_w - (int(half_w * porcentagem_corte)):half_w + (int(half_w * porcentagem_corte)) + 1,
    half_h - (int(half_h * porcentagem_corte)):half_h + (int(half_h * porcentagem_corte)) + 1] = 0

    shift_frq -= shift_frq_low
    imagem_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq)).real
    return imagem_filtrada


lista_corte = (0.05, 0.10, 0.15)
dir_imagens_ruidosas = './imagens_ruido_gaussiano/'
dir_imagens_filtro_passa_baixa = './imagens_filtro_passa_baixa/'

shutil.rmtree(dir_imagens_filtro_passa_baixa, ignore_errors=True)
os.mkdir(dir_imagens_filtro_passa_baixa)

lista_imagens = os.listdir(dir_imagens_ruidosas)
for porcentagem_corte in lista_corte:
    os.mkdir(dir_imagens_filtro_passa_baixa + str(porcentagem_corte))
    print(dir_imagens_filtro_passa_baixa + str(porcentagem_corte))
    for nome_imagem in lista_imagens:
        print(nome_imagem)
        imagem_ruidosa = img_as_float(imread(dir_imagens_ruidosas + nome_imagem, as_gray=True))
        linha, coluna = imagem_ruidosa.shape

        imagem_filtrada_passa_baixa = img_as_float(np.zeros(imagem_ruidosa.shape))

        imagem_filtrada_passa_baixa[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[:int(linha / 2), :int(coluna / 2)], porcentagem_corte)

        imagem_filtrada_passa_baixa[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[:int(linha / 2), int(coluna / 2):], porcentagem_corte)

        imagem_filtrada_passa_baixa[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[int(linha / 2):, :int(coluna / 2)], porcentagem_corte)

        imagem_filtrada_passa_baixa[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[int(linha / 2):, int(coluna / 2):], porcentagem_corte)


        imsave(dir_imagens_filtro_passa_baixa + str(porcentagem_corte) + '/' + nome_imagem, imagem_filtrada_passa_baixa)



print('FIM FILTRO PASSA BAIXA')