from builtins import print

import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.util import random_noise
from scipy import fftpack
import os
import shutil
import random
from matplotlib import pylab
import cv2 as cv


def filtro_passa_alta(imagem, porcentagem_corte):
    freq = fp.fft2(imagem)
    sfreq = fp.fftshift(freq)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)

    fcorte = int(half_w * porcentagem_corte)
    sfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0
    imagem_passa_alta = np.clip(fp.ifft2(fp.ifftshift(sfreq)).real, 0, 255)
    return imagem_passa_alta


def filtro_passa_baixa(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    shift_frq_low = np.copy(shift_frq)

    shift_frq_low[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0
    shift_frq -= shift_frq_low

    imagem_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq)).real
    return imagem_filtrada


def calcular_mse(img1, img2):
    return round(mse(img1, img2), 2)


def calcular_psnr(img1, img2):
    return round(psnr(img1, img2), 2)


def processar_filtros(imagem_original, imagem_filtrada):
    lista_psnr = []
    lista_mse = []
    l, c = imagem_original.shape
    q1_original = imagem_original[:int(l / 2), :int(c / 2)]  # QUADRANTE 1
    q2_original = imagem_original[:int(l / 2), int(c / 2):]  # QUADRANTE 2
    q3_original = imagem_original[int(l / 2):, int(c / 2):]  # QUADRANTE 3
    q4_original = imagem_original[int(l / 2):, :int(c / 2)]  # QUADRANTE 4

    q1_filtro = imagem_filtrada[:int(l / 2), :int(c / 2)]  # QUADRANTE 1
    q2_filtro = imagem_filtrada[:int(l / 2), int(c / 2):]  # QUADRANTE 2
    q3_filtro = imagem_filtrada[int(l / 2):, int(c / 2):]  # QUADRANTE 3
    q4_filtro = imagem_filtrada[int(l / 2):, :int(c / 2)]  # QUADRANTE 4


    lista_psnr.append(calcular_psnr(q1_original, q1_filtro))
    lista_psnr.append(calcular_psnr(q2_original, q2_filtro))
    lista_psnr.append(calcular_psnr(q3_original, q3_filtro))
    lista_psnr.append(calcular_psnr(q4_original, q4_filtro))

    lista_mse.append(calcular_mse(q1_original, q1_filtro))
    lista_mse.append(calcular_mse(q2_original, q2_filtro))
    lista_mse.append(calcular_mse(q3_original, q3_filtro))
    lista_mse.append(calcular_mse(q4_original, q4_filtro))
    return lista_psnr, lista_mse


def selecionar_maiores_valores(lista_valores):
    lista_valores_maximos = []
    lista_valores_maximos.append(np.max(lista_valores))
    lista_valores.remove(np.max(lista_valores))
    lista_valores_maximos.append(np.max(lista_valores))
    lista_valores.remove(np.max(lista_valores))
    lista_valores_maximos.append(np.max(lista_valores))
    lista_valores.remove(np.max(lista_valores))

    return sorted(lista_valores_maximos)


def processar_quadrante(lista_valores_quadrante, lista_psnr):
    maiores_valores = selecionar_maiores_valores(lista_valores_quadrante.copy())
    valor_medio_psnr = round(np.max(lista_valores_quadrante), 2)
    #print('Melhores PSNR: ' + str(maiores_valores))

    if valor_medio_psnr <= maiores_valores[0]:
        index_q1 = lista_valores_quadrante.index(maiores_valores[0])
        porcentagem_corte = lista_psnr[index_q1][0]
        #print('Melhor valor PSNR: ' + str(maiores_valores[0]))
        #print('Melhor porcentagem de corte: ' + str(porcentagem_corte))

    elif valor_medio_psnr <= maiores_valores[1]:
        index_q1 = lista_valores_quadrante.index(maiores_valores[1])
        porcentagem_corte = lista_psnr[index_q1][0]
        #print('Melhor valor PSNR: ' + str(maiores_valores[1]))
        #print('Melhor porcentagem de corte: ' + str(porcentagem_corte))

    else:
        index_q1 = lista_valores_quadrante.index(maiores_valores[2])
        porcentagem_corte = lista_psnr[index_q1][0]
        #print('Melhor valor PSNR: ' + str(maiores_valores[2]))
        #print('Melhor porcentagem de corte: ' + str(porcentagem_corte))
    return porcentagem_corte


def escolher_melhor_filtro(lista_psnr, lista_mse):

    lista_q1 = []
    lista_q2 = []
    lista_q3 = []
    lista_q4 = []

    for x in lista_psnr:
        lista_q1.append(x[1][0])
        lista_q2.append(x[1][1])
        lista_q3.append(x[1][2])
        lista_q4.append(x[1][3])

    #print('Processando Q1')
    corte_q1 = processar_quadrante(lista_q1, lista_psnr)
    #print()

    #print('Processando Q2')
    corte_q2 = processar_quadrante(lista_q2, lista_psnr)
    #print()

    #print('Processando Q3')
    corte_q3 = processar_quadrante(lista_q3, lista_psnr)
    #print()

    #print('Processando Q4')
    corte_q4 = processar_quadrante(lista_q4, lista_psnr)
    #print()
    return [corte_q1, corte_q2, corte_q3, corte_q4]


caminho_banco_imagem = '../banco_imagens/'
caminho_filtro_passa_alta = '../imagens_filtro_passa_alta/'
caminho_filtro_passa_baixa = '../imagens_filtro_passa_baixa/'
caminho_imagens_ruido = '../imagens_ruido_gaussiano/'

lista_psnr_passa_alta = []
lista_psnr_passa_baixa = []
lista_mse_passa_alta = []
lista_mse_passa_baixa= []


nome_arquivo_passa_alta = 'porcentagem_corte_passa_alta.txt'
nome_arquivo_passa_baixa = 'porcentagem_corte_passa_baixa.txt'
cabecalho_arquivo = 'Imagem;Q1;Q2;Q3;Q4;\n'
linha_arquivo_passa_alta = ''
linha_arquivo_passa_baixa = ''

lista_nome_imagem = os.listdir(caminho_banco_imagem)
for nome_imagem in lista_nome_imagem:
    linha_arquivo_passa_alta += nome_imagem + ';'
    linha_arquivo_passa_baixa += nome_imagem + ';'
    #print('Analisando imagem: ' + nome_imagem)
    imagem_original = imread(caminho_banco_imagem + nome_imagem, as_gray=True)
    imagem_ruido = imread(caminho_imagens_ruido + nome_imagem, as_gray=True)

    pylab.figure()
    linha = 4
    coluna = 6
    pylab.subplot(linha, coluna, 1)
    pylab.axis('off')
    pylab.imshow(imagem_original, cmap='gray')

    pylab.subplot(linha, coluna, 2)
    pylab.axis('off')
    pylab.imshow(imagem_ruido, cmap='gray')
    p = 7

    lista_corte_passa_alta = os.listdir(caminho_filtro_passa_alta)
    #print('Filtro Passa Alta')
    for corte_passa_alta in lista_corte_passa_alta:
        #print('corte_passa_alta: ' + str(corte_passa_alta))
        imagem_passa_alta = imread(caminho_filtro_passa_alta + corte_passa_alta + '/' + nome_imagem, as_gray=True)
        valores_psnr, valores_mse = processar_filtros(imagem_original, imagem_passa_alta)
        lista_psnr_passa_alta.append(np.array([corte_passa_alta, valores_psnr]))
        lista_mse_passa_alta.append([corte_passa_alta, valores_mse])

        pylab.subplot(linha, coluna, p)
        pylab.title(corte_passa_alta)
        pylab.axis('off')
        pylab.imshow(imagem_passa_alta, cmap='gray')
        p += 1

    melhores_porcentagens_passa_alta = escolher_melhor_filtro(lista_psnr_passa_alta, lista_mse_passa_alta)

    linha_arquivo_passa_alta += str(melhores_porcentagens_passa_alta[0]) + ';'
    linha_arquivo_passa_alta += str(melhores_porcentagens_passa_alta[1]) + ';'
    linha_arquivo_passa_alta += str(melhores_porcentagens_passa_alta[2]) + ';'
    linha_arquivo_passa_alta += str(melhores_porcentagens_passa_alta[3]) + ';'
    linha_arquivo_passa_alta += '\n'

    #print()
    #print('Filtro Passa Baixa')
    lista_corte_passa_baixa = os.listdir(caminho_filtro_passa_baixa)
    for corte_passa_baixa in lista_corte_passa_baixa:
        #print('corte_passa_baixa: ' + str(corte_passa_baixa))
        imagem_passa_baixa = imread(caminho_filtro_passa_baixa + corte_passa_baixa + '/' + nome_imagem, as_gray=True)
        valores_psnr, valores_mse = processar_filtros(imagem_original, imagem_passa_baixa)
        lista_psnr_passa_baixa.append([corte_passa_baixa, valores_psnr])
        lista_mse_passa_baixa.append([corte_passa_baixa, valores_mse])

        pylab.subplot(linha, coluna, p)
        pylab.title(corte_passa_baixa)
        pylab.axis('off')
        pylab.imshow(imagem_passa_baixa, cmap='gray')
        p += 1


    melhores_porcentagens_passa_baixa = escolher_melhor_filtro(lista_psnr_passa_baixa, lista_mse_passa_baixa)
    linha_arquivo_passa_baixa += str(melhores_porcentagens_passa_baixa[0]) + ';'
    linha_arquivo_passa_baixa += str(melhores_porcentagens_passa_baixa[1]) + ';'
    linha_arquivo_passa_baixa += str(melhores_porcentagens_passa_baixa[2]) + ';'
    linha_arquivo_passa_baixa += str(melhores_porcentagens_passa_baixa[3]) + ';'
    linha_arquivo_passa_baixa += '\n'

    nova_imagem_passa_alta = np.zeros(imagem_original.shape)
    l, c = imagem_ruido.shape
    q1_r = imagem_ruido[:int(l / 2), :int(c / 2)]  # QUADRANTE 1
    q2_r = imagem_ruido[:int(l / 2), int(c / 2):]  # QUADRANTE 2
    q3_r = imagem_ruido[int(l / 2):, int(c / 2):]  # QUADRANTE 3
    q4_r = imagem_ruido[int(l / 2):, :int(c / 2)]  # QUADRANTE 4


    #print(melhores_porcentagens_passa_alta[0])
    q1_pa = filtro_passa_alta(q1_r, float(melhores_porcentagens_passa_alta[0]))
    q2_pa = filtro_passa_alta(q2_r, float(melhores_porcentagens_passa_alta[1]))
    q3_pa = filtro_passa_alta(q3_r, float(melhores_porcentagens_passa_alta[2]))
    q4_pa = filtro_passa_alta(q4_r, float(melhores_porcentagens_passa_alta[3]))

    nova_imagem_passa_alta[:int(l / 2), :int(c / 2)] = q1_pa  # QUADRANTE 1
    nova_imagem_passa_alta[:int(l / 2), int(c / 2):] = q2_pa  # QUADRANTE 2
    nova_imagem_passa_alta[int(l / 2):, int(c / 2):] = q3_pa  # QUADRANTE 3
    nova_imagem_passa_alta[int(l / 2):, :int(c / 2)] = q4_pa  # QUADRANTE 4


    nova_imagem_passa_baixa = np.zeros(imagem_original.shape)
    q1_pb = filtro_passa_baixa(q1_r, float(melhores_porcentagens_passa_baixa[0]))
    q2_pb = filtro_passa_baixa(q2_r, float(melhores_porcentagens_passa_baixa[1]))
    q3_pb = filtro_passa_baixa(q3_r, float(melhores_porcentagens_passa_baixa[2]))
    q4_pb = filtro_passa_baixa(q4_r, float(melhores_porcentagens_passa_baixa[3]))
    nova_imagem_passa_baixa[:int(l / 2), :int(c / 2)] = q1_pb  # QUADRANTE 1
    nova_imagem_passa_baixa[:int(l / 2), int(c / 2):] = q2_pb  # QUADRANTE 2
    nova_imagem_passa_baixa[int(l / 2):, int(c / 2):] = q3_pb  # QUADRANTE 3
    nova_imagem_passa_baixa[int(l / 2):, :int(c / 2)] = q4_pb  # QUADRANTE 4


    pylab.subplot(linha, coluna, p)
    titulo_pa = 'q1:' + melhores_porcentagens_passa_alta[0] + '|'
    titulo_pa += 'q2:' + melhores_porcentagens_passa_alta[1] + '|'
    titulo_pa += 'q3:' + melhores_porcentagens_passa_alta[2] + '|'
    titulo_pa += 'q4:' + melhores_porcentagens_passa_alta[3]
    pylab.title(titulo_pa)
    pylab.axis('off')
    pylab.imshow(nova_imagem_passa_alta, cmap='gray')
    p += 2

    pylab.subplot(linha, coluna, p)
    titulo_pb = 'q1:' + melhores_porcentagens_passa_baixa[0] + '|'
    titulo_pb += 'q2:' + melhores_porcentagens_passa_baixa[1] + '|'
    titulo_pb += 'q3:' + melhores_porcentagens_passa_baixa[2] + '|'
    titulo_pb += 'q4:' + melhores_porcentagens_passa_baixa[3]
    pylab.title(titulo_pb)
    pylab.axis('off')
    pylab.imshow(nova_imagem_passa_baixa, cmap='gray')
    p += 1

    imagem_final_pb_pa = nova_imagem_passa_baixa + nova_imagem_passa_alta
    pylab.subplot(linha, coluna, p)
    pylab.title('Passa baixa + Passa Alta')
    pylab.axis('off')
    pylab.imshow(imagem_final_pb_pa, cmap='gray')
    p += 1

    #imagem_final_pa_pb = nova_imagem_passa_baixa - nova_imagem_passa_alta
    #pylab.subplot(linha, coluna, p)
    #pylab.title('Passa Alta * Passa baixa')
    #pylab.axis('off')
    #pylab.imshow(imagem_final_pa_pb, cmap='gray')
    #p += 1

    pylab.show()
    pylab.close()

    #pylab.imshow(nova_imagem_passa_baixa, cmap='gray'), pylab.show()
    #pylab.imshow(nova_imagem_passa_alta, cmap='gray'), pylab.show()


print(nome_arquivo_passa_alta)
print(cabecalho_arquivo)
print(linha_arquivo_passa_alta)
print()
print(nome_arquivo_passa_alta)
print(cabecalho_arquivo)
print(linha_arquivo_passa_baixa)




print('FIM TESTE ESCOLHER MELHOR FILTRO')