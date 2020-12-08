import numpy as np
from skimage.io import imread
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr

import os

import codecs

def calcular_mse(img_filtrada, img_ruidosa):
    return round(mse(img_filtrada, img_ruidosa), 2)


def calcular_psnr(img_ruidosa, img_filtrada):
    return round(psnr(img_ruidosa, img_filtrada), 2)


def salvar_arquivo(diretorio_arquivo, nome_arquivo, conteudo_arquivo):
    arquivo_mse = codecs.open(diretorio_arquivo + nome_arquivo, 'w', 'utf-8')
    arquivo_mse.write(conteudo_arquivo)
    arquivo_mse.close()


lista_corte = (0.05, 0.07, 0.10, 0.13, 0.15)
caminho_imagem_ruidosa = './imagens_ruido_gaussiano/'
lista_caminho_imagem_filtrada = (['./imagens_filtro_passa_alta/', '_alta_'], ['./imagens_filtro_passa_baixa/', '_baixa_'])

dir_metricas = './metricas/'

for caminho_imagem_filtrada in lista_caminho_imagem_filtrada:

    dir_imagem_filtrada = caminho_imagem_filtrada[0]
    sulf_arquivo = caminho_imagem_filtrada[1]

    lista_porcentagem_corte = os.listdir(dir_imagem_filtrada)
    for porncetagem_corte in lista_porcentagem_corte:

        cabecalho_arquivo = 'Imagem;Q1;Q2;Q3;Q4;\n'
        linha_arquivo_mse = cabecalho_arquivo + ''
        linha_arquivo_psnr = cabecalho_arquivo + ''
        nome_arquivo_mse = 'mse_filtro_passa'+ sulf_arquivo + str(porncetagem_corte) + '.txt'
        nome_arquivo_psnr = 'psnr_filtro_passa' + sulf_arquivo + str(porncetagem_corte) + '.txt'
        print('Criando arquivos' + nome_arquivo_mse + ', ' + nome_arquivo_psnr)


        lista_imagens_ruidosas = os.listdir(caminho_imagem_ruidosa)
        for nome_imagem in lista_imagens_ruidosas:
            # ABRIR IMAGENS RUIDOSAS
            imagem_ruidosa = imread(caminho_imagem_ruidosa + nome_imagem)
            linha, coluna = imagem_ruidosa.shape
            q1_ruidoso = imagem_ruidosa[:int(linha / 2), :int(coluna / 2)]
            q2_ruidoso = imagem_ruidosa[:int(linha / 2), int(coluna / 2):]
            q3_ruidoso = imagem_ruidosa[int(linha / 2), :int(coluna / 2)]
            q4_ruidoso = imagem_ruidosa[int(linha / 2), int(coluna / 2):]

            #ABRIR IMAGENS FILTRADAS
            imagem_filtrada = imread(dir_imagem_filtrada + '/' + str(porncetagem_corte) + '/' + nome_imagem)
            q1_filtrado = imagem_filtrada[:int(linha / 2), :int(coluna / 2)]
            q2_filtrado = imagem_filtrada[:int(linha / 2), int(coluna / 2):]
            q3_filtrado = imagem_filtrada[int(linha / 2), :int(coluna / 2)]
            q4_filtrado = imagem_filtrada[int(linha / 2), int(coluna / 2):]

            #CALCULAR MSE
            q1_mse = calcular_mse(q1_ruidoso, q1_filtrado)
            q2_mse = calcular_mse(q2_ruidoso, q2_filtrado)
            q3_mse = calcular_mse(q3_ruidoso, q3_filtrado)
            q4_mse = calcular_mse(q4_ruidoso, q4_filtrado)

            # CALCULAR PSNR
            q1_psnr = calcular_psnr(q1_ruidoso, q1_filtrado)
            q2_psnr = calcular_psnr(q2_ruidoso, q2_filtrado)
            q3_psnr = calcular_psnr(q3_ruidoso, q3_filtrado)
            q4_psnr = calcular_psnr(q4_ruidoso, q4_filtrado)

            linha_arquivo_mse += nome_imagem + ';' + str(q1_mse) + ';' + str(q2_mse) + ';' + str(q3_mse) + ';' + str(q4_mse)
            linha_arquivo_mse += '\n'

            linha_arquivo_psnr += nome_imagem + ';' + str(q1_psnr) + ';' + str(q2_psnr) + ';' + str(q3_psnr) + ';' + str(q4_psnr)
            linha_arquivo_psnr += '\n'

        salvar_arquivo(dir_metricas, nome_arquivo_mse, linha_arquivo_mse)
        salvar_arquivo(dir_metricas, nome_arquivo_psnr, linha_arquivo_psnr)

print('FIM MÉTRICA MSE')
