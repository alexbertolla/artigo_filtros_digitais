import numpy as np
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import codecs

def calcular_psnr(img_ruidosa, img_filtrada):
    return round(psnr(img_ruidosa, img_filtrada), 2)

lista_corte = (0.05, 0.07, 0.10, 0.13, 0.15)
#lista_corte = (0.10, 0.15, 0.20, 1)
caminho_imagem_original = './banco_imagens/'
caminho_imagem_ruidosa = './imagens_ruido_gaussiano/'
caminho_imagem_filtrada = './imagens_filtro_passa_alta/'
dir_metricas = './metricas/'

nome_arquivo = 'psnr_filtro_passa_alta.txt'
cabecalho_arquivo = 'Imagem;Corte 5%;Corte 7%;Corte 10%;Corte 13%;Corte 15%;\n'
linha_arquivo = ''

lista_imagens_ruidosas = os.listdir(caminho_imagem_ruidosa)
for nome_imagem in lista_imagens_ruidosas:
    linha_arquivo += nome_imagem + ';'

    imagem_original = imread(caminho_imagem_original + nome_imagem)
    imagem_ruidosa = imread(caminho_imagem_ruidosa + nome_imagem)
    linha, coluna = imagem_ruidosa.shape

    q1_original = imagem_original[:int(linha / 2), :int(coluna / 2)]
    q2_original = imagem_original[:int(linha / 2), int(coluna / 2):]
    q3_original = imagem_original[int(linha / 2), :int(coluna / 2)]
    q4_original = imagem_original[int(linha / 2), int(coluna / 2):]

    q1_ruidoso = imagem_ruidosa[:int(linha / 2), :int(coluna / 2)]
    q2_ruidoso = imagem_ruidosa[:int(linha / 2), int(coluna / 2):]
    q3_ruidoso = imagem_ruidosa[int(linha / 2), :int(coluna / 2)]
    q4_ruidoso = imagem_ruidosa[int(linha / 2), int(coluna / 2):]


    for valor_corte in lista_corte:
        imagem_filtrada = imread(caminho_imagem_filtrada + '/' + str(valor_corte) + '/' + nome_imagem)
        q1_filtrado = imagem_filtrada[:int(linha / 2), :int(coluna / 2)]
        q2_filtrado = imagem_filtrada[:int(linha / 2), int(coluna / 2):]
        q3_filtrado = imagem_filtrada[int(linha / 2), :int(coluna / 2)]
        q4_filtrado = imagem_filtrada[int(linha / 2), int(coluna / 2):]

        array_psnr = []
        array_psnr.append(calcular_psnr(q1_original, q1_filtrado))
        array_psnr.append(calcular_psnr(q2_original, q2_filtrado))
        array_psnr.append(calcular_psnr(q3_original, q3_filtrado))
        array_psnr.append(calcular_psnr(q4_original, q4_filtrado))
        linha_arquivo += str(round(np.median(array_psnr), 2)) + ';'

    linha_arquivo += '\n'

arquivo_mse = codecs.open(dir_metricas + nome_arquivo, 'w', 'utf-8')
arquivo_mse.write(cabecalho_arquivo)
arquivo_mse.write(linha_arquivo)
arquivo_mse.close()
#print(cabecalho_arquivo)
#print(linha_arquivo)



print('FIM MÉTRICA PSNR')
