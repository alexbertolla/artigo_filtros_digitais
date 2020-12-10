import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from scipy import fftpack
import os
import shutil

def filtro_passa_baixa(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    shift_frq_low = np.copy(shift_frq)

    shift_frq_low[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0
    shift_frq -= shift_frq_low

    imagem_filtrada = np.clip(fp.ifft2(fftpack.ifftshift(shift_frq)).real, 0, 1)
    return imagem_filtrada


lista_porcentagem_corte = ([0.05, 0.07, 0.10, 0.13, 0.15, 0.20])
dir_imagens_ruidosas = './imagens_ruido_gaussiano/'
dir_imagens_filtro_passa_baixa = './imagens_filtro_passa_baixa/'

shutil.rmtree(dir_imagens_filtro_passa_baixa, ignore_errors=True)
os.mkdir(dir_imagens_filtro_passa_baixa)

lista_imagens = os.listdir(dir_imagens_ruidosas)
total_imagens = len(lista_imagens)
total_filtros = len(lista_porcentagem_corte)
aux_total_filtros = 1


for porcentagem_corte in lista_porcentagem_corte:
    os.mkdir(dir_imagens_filtro_passa_baixa + str(porcentagem_corte))
    print('diretório ' + dir_imagens_filtro_passa_baixa + str(porcentagem_corte) + ' criado ')
    aux_total_imagens = 1
    for nome_imagem in lista_imagens:
        print(str(aux_total_filtros) + ' de ' + str(total_filtros) + ' filtros' + ' | ' + str(aux_total_imagens) + ' de ' + str(total_imagens) + ' imagens')
        aux_total_imagens += 1
        imagem_ruido = img_as_float(imread(dir_imagens_ruidosas + nome_imagem, as_gray=True))

        #############SEPARA QUADRANTES#############
        linha, coluna = imagem_ruido.shape
        q1_ruido = imagem_ruido[:int(linha / 2), :int(coluna / 2)]  # QUADRANTE 1
        q2_ruido = imagem_ruido[:int(linha / 2), int(coluna / 2):]  # QUADRANTE 2
        q3_ruido = imagem_ruido[int(linha / 2):, int(coluna / 2):]  # QUADRANTE 3
        q4_ruido = imagem_ruido[int(linha / 2):, :int(coluna / 2)]  # QUADRANTE 4

        #############FIM SEPARA QUADRANTES#############




        q1_filtro = filtro_passa_baixa(q1_ruido, porcentagem_corte)
        q2_filtro = filtro_passa_baixa(q2_ruido, porcentagem_corte)
        q3_filtro = filtro_passa_baixa(q3_ruido, porcentagem_corte)
        q4_filtro = filtro_passa_baixa(q4_ruido, porcentagem_corte)

        imagem_filtrada = np.zeros(imagem_ruido.shape)
        imagem_filtrada[:int(linha / 2), :int(coluna / 2)] = q1_filtro  # QUADRANTE 4
        imagem_filtrada[:int(linha / 2), int(coluna / 2):] = q2_filtro  # QUADRANTE 1
        imagem_filtrada[int(linha / 2):, int(coluna / 2):] = q3_filtro  # QUADRANTE 2
        imagem_filtrada[int(linha / 2):, :int(coluna / 2)] = q4_filtro  # QUADRANTE 3


        imagem_filtrada = img_as_ubyte(imagem_filtrada)

        imsave(dir_imagens_filtro_passa_baixa + str(porcentagem_corte) + '/' + nome_imagem, imagem_filtrada)
        #print(dir_imagens_filtro_passa_baixa + str(porcentagem_corte) + '/' + nome_imagem)
    aux_total_filtros += 1



print('FIM FILTRO PASSA BAIXA')