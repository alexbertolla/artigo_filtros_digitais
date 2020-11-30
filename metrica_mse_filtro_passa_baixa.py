import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte, img_as_float32
from skimage.io import imread, imsave
from skimage.metrics import mean_squared_error as mse
from scipy import fftpack
from matplotlib import pylab
import os
from os import path
import shutil
import codecs

def calcular_mse(img_filtrada, img_ruidosa):
    return round(mse(img_filtrada, img_ruidosa), 2)

corte_array = (20, 25, 30, 35)


dir_imagens_orignais = './imagens_originais/'
dir_metricas = './metricas/'
dir_array = (
#    ['./imagens_originais/', './imagens_originais_filtro_passa_baixa/', 'Imagens Originais/'],
    ['./imagens_ruido_gaussiano/', './imagens_ruido_gaussiano_filtro_passa_baixa/', 'mse_ruido_gaussiano_filtro_passa_baixa.txt'],
    ['./imagens_ruido_sal_e_pimenta/', './imagens_ruido_sal_e_pimenta_filtro_passa_baixa/', 'mse_ruido_impulsivo_filtro_passa_baixa.txt'],
    ['./imagens_ruido_spekle/', './imagens_ruido_spekle_filtro_passa_baixa/', 'mse_ruido_spekle_filtro_passa_baixa.txt'],
)



for diretorio in dir_array:
    linha_arquivo = ''
    dir_imagem_ruidosa = diretorio[0]
    dir_imagem_filtrada = diretorio[1]
    nome_arquivo = diretorio[2]
    print(nome_arquivo)

    #arquivo_mse_filtro_wiener.write(cabecalho_mse)

    lista_imagens = os.listdir(dir_imagens_orignais)
    for nome_imagem in lista_imagens:
        cabecalho_arquivo = 'Imagem;'
        linha_arquivo += nome_imagem + ';'
        imagem_original = imread(dir_imagens_orignais + nome_imagem)
        imagem_ruidosa = imread(dir_imagem_ruidosa + nome_imagem)

        linha, coluna = imagem_ruidosa.shape
        imagem_ruidosa_q1 = imagem_ruidosa[:int(linha / 2), :int(coluna / 2)]
        imagem_ruidosa_q2 = imagem_ruidosa[:int(linha / 2), int(coluna / 2):]
        imagem_ruidosa_q3 = imagem_ruidosa[int(linha / 2), :int(coluna / 2)]
        imagem_ruidosa_q4 = imagem_ruidosa[int(linha / 2), int(coluna / 2):]

        for valor_corte in corte_array:
            cabecalho_arquivo += str(valor_corte) + ';'
            imagem_filtrada = imread(dir_imagem_filtrada +'/'+ str(valor_corte) +'/'+ nome_imagem)
            imagem_filtrada_q1 = imagem_filtrada[:int(linha / 2), :int(coluna / 2)]
            imagem_filtrada_q2 = imagem_filtrada[:int(linha / 2), int(coluna / 2):]
            imagem_filtrada_q3 = imagem_filtrada[int(linha / 2), :int(coluna / 2)]
            imagem_filtrada_q4 = imagem_filtrada[int(linha / 2), int(coluna / 2):]

            mse_q1 = calcular_mse(imagem_filtrada_q1, imagem_ruidosa_q1)
            mse_q2 = calcular_mse(imagem_filtrada_q2, imagem_ruidosa_q2)
            mse_q3 = calcular_mse(imagem_filtrada_q3, imagem_ruidosa_q3)
            mse_q4 = calcular_mse(imagem_filtrada_q4, imagem_ruidosa_q4)

            media_mse = round((mse_q1 + mse_q2 + mse_q3 + mse_q4) / 4, 2)
            linha_arquivo += str(media_mse) + ';'

        cabecalho_arquivo += '\n'
        linha_arquivo += '\n'

    arquivo_mse = codecs.open(dir_metricas + nome_arquivo, 'w', 'utf-8')
    arquivo_mse.write(cabecalho_arquivo)
    arquivo_mse.write(linha_arquivo)
    arquivo_mse.close()
    #print(nome_arquivo)
    #print(cabecalho_arquivo)
    #print(linha_arquivo)

print('FIM MÉTRICA MSE')
