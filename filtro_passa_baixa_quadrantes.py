import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte, img_as_float32
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import fftpack
from matplotlib import pylab
import os
from os import path
import shutil


def calcular_psnr(imagem_ruidosa, imagem_filtrada):
    return round(psnr(imagem_ruidosa, imagem_filtrada), 2)

def gerar_spectro(transformata_imagem):
    shift_frq = fftpack.fftshift(transformata_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

def plotar_imagem(linha, coluna, posicao, imagem, titulo):
    pylab.subplot(linha, coluna, posicao)
    pylab.axis('off')
    pylab.title(titulo)
    pylab.imshow(imagem, cmap='gray')


lista_corte = (20, 25, 30, 35)
dir_array = (
    #['./imagens_originais/', './imagens_originais_filtro_passa_baixa/', 'Imagens Originais/'],
    ['./imagens_ruido_gaussiano/', './imagens_ruido_gaussiano_filtro_passa_baixa/', 'Imagens Ruido Gaussiano/'],
    ['./imagens_ruido_sal_e_pimenta/', './imagens_ruido_sal_e_pimenta_filtro_passa_baixa/', 'Imagens Ruido Impulsivo/'],
    ['./imagens_ruido_spekle/', './imagens_ruido_spekle_filtro_passa_baixa/', 'Imagens Ruido Spekle/'],
)

for diretorio in dir_array:
    diretorio_origem = diretorio[0]
    diretorio_destino = diretorio[1]
    titulo = diretorio[2]
    lista_imagens = os.listdir(diretorio_origem)
    for nome_imagem in lista_imagens:
        imagem_original = img_as_float(imread(diretorio_origem + nome_imagem, as_gray=True))
        imagem_filtrada = img_as_float(np.zeros(imagem_original.shape))

        linha, coluna = imagem_original.shape

        quadrante_1 = img_as_float(imagem_original[:int(linha / 2), :int(coluna / 2)])
        quadrante_2 = img_as_float(imagem_original[:int(linha / 2), int(coluna / 2):])
        quadrante_3 = img_as_float(imagem_original[int(linha / 2):, :int(coluna / 2)])
        quadrante_4 = img_as_float(imagem_original[int(linha / 2):, int(coluna / 2):])

        plt_linha = 1
        plt_coluna = len(lista_corte)
        plt_posicao = 1

        pylab.figure()

        for corte in lista_corte:
            discrete_transform_q1 = fp.fft2(quadrante_1)
            (w_q1, h_q1) = discrete_transform_q1.shape
            half_w_q1, half_h_q1 = int(w_q1 / 2), int(h_q1 / 2)
            spectro_q1 = gerar_spectro(discrete_transform_q1)

            shift_frq_q1 = fftpack.fftshift(discrete_transform_q1)
            shift_frq_low_q1 = np.copy(shift_frq_q1)
            shift_frq_low_q1[half_w_q1 - corte:half_w_q1 + corte + 1, half_h_q1 - corte:half_h_q1 + corte + 1] = 0
            shift_frq_q1 -= shift_frq_low_q1
            q1_filtrado = img_as_float(fp.ifft2(fftpack.ifftshift(shift_frq_q1)).real)
            valor_psnr = calcular_psnr(quadrante_1, q1_filtrado)

            plotar_imagem(plt_linha, plt_coluna, plt_posicao, q1_filtrado, str(valor_psnr) + '|' + str(corte))
            plt_posicao += 1
        pylab.show()
    pylab.close()




print('FIM FILTRO PASSA BAIXA QUADRANTES')