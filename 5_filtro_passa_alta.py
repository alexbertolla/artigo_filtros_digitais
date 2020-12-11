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

def filtro_passa_alta(imagem, porcentagem_corte):
    freq = fp.fft2(imagem)
    sfreq = fp.fftshift(freq)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)
    sfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0
    imagem_passa_alta = np.clip(fp.ifft2(fp.ifftshift(sfreq)).real, 0, 1)
    return imagem_passa_alta

def processar_filtro(imagem_ruido):
    #############SEPARA QUADRANTES#############
    linha, coluna = imagem_ruido.shape
    q1_ruido = imagem_ruido[:int(linha / 2), :int(coluna / 2)]  # QUADRANTE 1
    q2_ruido = imagem_ruido[:int(linha / 2), int(coluna / 2):]  # QUADRANTE 2
    q3_ruido = imagem_ruido[int(linha / 2):, int(coluna / 2):]  # QUADRANTE 3
    q4_ruido = imagem_ruido[int(linha / 2):, :int(coluna / 2)]  # QUADRANTE 4
    #############FIM SEPARA QUADRANTES#############

    q1_filtro = filtro_passa_alta(q1_ruido, porcentagem_corte)
    q2_filtro = filtro_passa_alta(q2_ruido, porcentagem_corte)
    q3_filtro = filtro_passa_alta(q3_ruido, porcentagem_corte)
    q4_filtro = filtro_passa_alta(q4_ruido, porcentagem_corte)

    imagem_filtrada = np.zeros(imagem_ruido.shape)
    imagem_filtrada[:int(linha / 2), :int(coluna / 2)] = q1_filtro  # QUADRANTE 1
    imagem_filtrada[:int(linha / 2), int(coluna / 2):] = q2_filtro  # QUADRANTE 2
    imagem_filtrada[int(linha / 2):, int(coluna / 2):] = q3_filtro  # QUADRANTE 3
    imagem_filtrada[int(linha / 2):, :int(coluna / 2)] = q4_filtro  # QUADRANTE 4
    return imagem_filtrada

lista_porcentagem_corte = ([0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
dir_imagens_ruido_gaussiano = './imagens_ruido_gaussiano/'
dir_imagens_ruido_impulsivo = './imagens_ruido_impulsivo/'
dir_imagens_filtro_passa_alta = './imagens_filtro_passa_alta/'
dir_destino_ruido_gaussiano = dir_imagens_filtro_passa_alta + 'ruido_gaussiano/'
dir_destino_ruido_impulsivo = dir_imagens_filtro_passa_alta + 'ruido_impulsivo/'


shutil.rmtree(dir_imagens_filtro_passa_alta, ignore_errors=True)
os.mkdir(dir_imagens_filtro_passa_alta)
os.mkdir(dir_destino_ruido_gaussiano)
os.mkdir(dir_destino_ruido_impulsivo)



lista_imagens = os.listdir(dir_imagens_ruido_gaussiano)
total_imagens = len(lista_imagens)
total_filtros = len(lista_porcentagem_corte)
aux_total_filtros = 1


for porcentagem_corte in lista_porcentagem_corte:
    os.mkdir(dir_destino_ruido_gaussiano + str(porcentagem_corte))
    os.mkdir(dir_destino_ruido_impulsivo + str(porcentagem_corte))
    print('diretório ' + dir_imagens_filtro_passa_alta + str(porcentagem_corte) + ' criado ')
    aux_total_imagens = 1

    for nome_imagem in lista_imagens:
        print(str(aux_total_filtros) + ' de ' + str(total_filtros) + ' filtros' + ' | ' + str(aux_total_imagens) + ' de ' + str(total_imagens) + ' imagens')
        aux_total_imagens += 1

        imagem_ruido_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano + nome_imagem, as_gray=True))
        imagem_ruido_gaussiano_filtrada = processar_filtro(imagem_ruido_gaussiano)
        imagem_ruido_gaussiano_filtrada = img_as_ubyte(imagem_ruido_gaussiano_filtrada)
        imsave(dir_destino_ruido_gaussiano + str(porcentagem_corte) + '/' + nome_imagem, imagem_ruido_gaussiano_filtrada)

        imagem_ruido_impulsivo = img_as_float(imread(dir_imagens_ruido_impulsivo + nome_imagem, as_gray=True))
        imagem_ruido_impulsivo_filtrada = processar_filtro(imagem_ruido_impulsivo)
        imagem_ruido_impulsivo_filtrada = img_as_ubyte(imagem_ruido_impulsivo_filtrada)
        imsave(dir_destino_ruido_impulsivo + str(porcentagem_corte) + '/' + nome_imagem, imagem_ruido_impulsivo_filtrada)
    aux_total_filtros += 1




print('FIM FILTRO PASSA ALTA')