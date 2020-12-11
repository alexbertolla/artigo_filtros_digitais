from skimage.io import imread
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
import shutil
import codecs

def calcular_mse(img1, img2):
    return round(mse(img1, img2), 2)


def calcular_psnr(img1, img2):
    return round(psnr(img1, img2), 2)


def salvar_arquivo(diretorio_arquivo, nome_arquivo, conteudo_arquivo):
    arquivo_mse = codecs.open(diretorio_arquivo + nome_arquivo, 'w', 'utf-8')
    arquivo_mse.write(conteudo_arquivo)
    arquivo_mse.close()


def montar_arquivo(imagem_original, imagem_ruidosa, imagem_filtrada, conteudo_psnr, conteudo_mse):
    linha, coluna = imagem_ruidosa.shape
    q1_ruidoso = imagem_ruidosa[:int(linha / 2), :int(coluna / 2)]  # QUADRANTE 1
    q2_ruidoso = imagem_ruidosa[:int(linha / 2), int(coluna / 2):]  # QUADRANTE 2
    q3_ruidoso = imagem_ruidosa[int(linha / 2):, int(coluna / 2):]  # QUADRANTE 3
    q4_ruidoso = imagem_ruidosa[int(linha / 2):, :int(coluna / 2)]  # QUADRANTE 4

    q1_filtrado = imagem_filtrada[:int(linha / 2), :int(coluna / 2)]  # QUADRANTE 1
    q2_filtrado = imagem_filtrada[:int(linha / 2), int(coluna / 2):]  # QUADRANTE 2
    q3_filtrado = imagem_filtrada[int(linha / 2):, int(coluna / 2):]  # QUADRANTE 3
    q4_filtrado = imagem_filtrada[int(linha / 2):, :int(coluna / 2)]  # QUADRANTE 4

    q1_original = imagem_original[:int(linha / 2), :int(coluna / 2)]  # QUADRANTE 1
    q2_original = imagem_original[:int(linha / 2), int(coluna / 2):]  # QUADRANTE 2
    q3_original = imagem_original[int(linha / 2):, int(coluna / 2):]  # QUADRANTE 3
    q4_original = imagem_original[int(linha / 2):, :int(coluna / 2)]  # QUADRANTE 4

    # CALCULAR MSE
    q1_mse = calcular_mse(q1_original, q1_filtrado)
    q2_mse = calcular_mse(q2_original, q2_filtrado)
    q3_mse = calcular_mse(q3_original, q3_filtrado)
    q4_mse = calcular_mse(q4_original, q4_filtrado)

    # CALCULAR PSNR
    q1_psnr = calcular_psnr(q1_original, q1_filtrado)
    q2_psnr = calcular_psnr(q2_original, q2_filtrado)
    q3_psnr = calcular_psnr(q3_original, q3_filtrado)
    q4_psnr = calcular_psnr(q4_original, q4_filtrado)

    conteudo_mse += nome_imagem + ';' + str(q1_mse) + ';' + str(q2_mse) + ';' + str(q3_mse) + ';' + str(q4_mse)
    conteudo_mse += '\n'

    conteudo_psnr += nome_imagem + ';' + str(q1_psnr) + ';' + str(q2_psnr) + ';' + str(q3_psnr) + ';' + str(q4_psnr)
    conteudo_psnr += '\n'

    return conteudo_psnr, conteudo_mse







lista_corte = (0.05, 0.07, 0.10, 0.13, 0.15)
caminho_imagem_original = './banco_imagens/'
caminho_imagem_ruido_gaussiano = './imagens_ruido_gaussiano/'
caminho_imagem_ruido_impulsivo = './imagens_ruido_impulsivo/'
lista_caminho_imagem_filtrada = ([['./imagens_filtro_passa_alta/ruido_gaussiano',
                                   './imagens_filtro_passa_alta/ruido_impulsivo'], '_alta_'],
                                 [['./imagens_filtro_passa_baixa/ruido_gaussiano',
                                   './imagens_filtro_passa_baixa/ruido_impulsivo'], '_baixa_'])

dir_metricas = './metricas/'
shutil.rmtree(dir_metricas, ignore_errors=True)
os.mkdir(dir_metricas)
os.mkdir(dir_metricas + 'ruido_gaussiano/')
os.mkdir(dir_metricas + 'ruido_impulsivo/')
os.mkdir(dir_metricas + 'ruido_gaussiano/' + 'mse')
os.mkdir(dir_metricas + 'ruido_impulsivo/' + 'mse')
os.mkdir(dir_metricas + 'ruido_gaussiano/' + 'psnr')
os.mkdir(dir_metricas + 'ruido_impulsivo/' + 'psnr')


for caminho_imagem_filtrada in lista_caminho_imagem_filtrada:

    dir_imagem_filtro_gaussiano = caminho_imagem_filtrada[0][0]
    dir_imagem_filtro_impulsivo = caminho_imagem_filtrada[0][1]
    sulf_arquivo = caminho_imagem_filtrada[1]

    lista_porcentagem_corte = os.listdir(dir_imagem_filtro_gaussiano)
    for porncetagem_corte in lista_porcentagem_corte:



        cabecalho_arquivo = 'Imagem;Q1;Q2;Q3;Q4;\n'
        linha_arquivo_mse_gaussiano = cabecalho_arquivo + ''
        linha_arquivo_psnr_gaussiano = cabecalho_arquivo + ''
        linha_arquivo_mse_impulsivo = cabecalho_arquivo + ''
        linha_arquivo_psnr_impulsivo = cabecalho_arquivo + ''
        nome_arquivo_mse = 'mse_filtro_passa'+ sulf_arquivo + str(porncetagem_corte) + '.txt'
        nome_arquivo_psnr = 'psnr_filtro_passa' + sulf_arquivo + str(porncetagem_corte) + '.txt'
        print('Criando arquivos' + nome_arquivo_mse + ', ' + nome_arquivo_psnr)


        lista_imagens_ruidosas = os.listdir(caminho_imagem_ruido_gaussiano)
        for nome_imagem in lista_imagens_ruidosas:
            imagem_ruido_gaussiano = imread(caminho_imagem_ruido_gaussiano + nome_imagem)
            imagem_ruido_impulsivo = imread(caminho_imagem_ruido_impulsivo + nome_imagem)

            imagem_filtro_gaussiano = imread(dir_imagem_filtro_gaussiano + '/' + str(porncetagem_corte) + '/' + nome_imagem)
            imagem_filtro_impulsivo = imread(dir_imagem_filtro_impulsivo + '/' + str(porncetagem_corte) + '/' + nome_imagem)
            imagem_original = imread(caminho_imagem_original + '/' + nome_imagem)


            linha_arquivo_psnr_gaussiano, linha_arquivo_mse_gaussiano = montar_arquivo(imagem_original,
                                                                                       imagem_ruido_gaussiano,
                                                                                       imagem_filtro_gaussiano,
                                                                                       linha_arquivo_psnr_gaussiano,
                                                                                       linha_arquivo_mse_gaussiano)


            linha_arquivo_psnr_impulsivo, linha_arquivo_mse_impulsivo = montar_arquivo(imagem_original,
                                                                                       imagem_ruido_impulsivo,
                                                                                       imagem_filtro_impulsivo,
                                                                                       linha_arquivo_psnr_impulsivo,
                                                                                       linha_arquivo_mse_impulsivo)

        salvar_arquivo(dir_metricas + 'ruido_gaussiano/' + 'psnr/', nome_arquivo_psnr, linha_arquivo_psnr_gaussiano)
        salvar_arquivo(dir_metricas + 'ruido_gaussiano/' + 'mse/', nome_arquivo_mse, linha_arquivo_mse_gaussiano)

        salvar_arquivo(dir_metricas + 'ruido_impulsivo/' + 'psnr/', nome_arquivo_psnr, linha_arquivo_psnr_impulsivo)
        salvar_arquivo(dir_metricas + 'ruido_impulsivo/' + 'mse/', nome_arquivo_mse, linha_arquivo_mse_impulsivo)

print('FIM CALCULAR MÉTRICAS MSE & PSNR')
