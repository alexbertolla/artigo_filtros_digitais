from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from sporco.metric import isnr
from matplotlib import pylab
import os
import codecs


def calcular_isnr(imagem_original, imagem_ruidosa, imagem_filtrada):
    return isnr(imagem_original, imagem_ruidosa, imagem_filtrada)


def calcular_mse(imagem_ruidosa, imagem_filtrada):
    return mean_squared_error(imagem_ruidosa, imagem_filtrada)

def calcular_ssim(imagem_original, imagem_filtrada):
    return structural_similarity(imagem_original, imagem_filtrada, multichannel=False)




dir_imagens_originais = 'imagens_originais'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'
dir_imagens_ruido_nao_estacionario = 'imagens_ruido_nao_estacionario'


dir_imagens_ruido_gaussiano_filtro_gaussiano = 'imagens_ruido_gaussiano_filtro_gaussiano'
dir_imagens_ruido_gaussiano_filtro_mediana = 'imagens_ruido_gaussiano_filtro_mediana'
dir_imagens_ruido_gaussiano_filtro_nlm = 'imagens_ruido_gaussiano_filtro_nlm'
dir_imagens_ruido_gaussiano_filtro_wiener = 'imagens_ruido_gaussiano_filtro_wiener'


dir_imagens_ruido_spekle_filtro_gaussiano = 'imagens_ruido_spekle_filtro_gaussiano'
dir_imagens_ruido_spekle_filtro_mediana = 'imagens_ruido_spekle_filtro_mediana'
dir_imagens_ruido_spekle_filtro_nlm = 'imagens_ruido_spekle_filtro_nlm'
dir_imagens_ruido_spekle_filtro_wiener = 'imagens_ruido_spekle_filtro_wiener'


dir_imagens_ruido_sal_e_pimenta_filtro_gaussiano = 'imagens_ruido_sal_e_pimenta_filtro_gaussiano'
dir_imagens_ruido_sal_e_pimenta_filtro_mediana = 'imagens_ruido_sal_e_pimenta_filtro_mediana'
dir_imagens_ruido_sal_e_pimenta_filtro_nlm = 'imagens_ruido_sal_e_pimenta_filtro_nlm'
dir_imagens_ruido_sal_e_pimenta_filtro_wiener = 'imagens_ruido_sal_e_pimenta_filtro_wiener'


cabecalho_mse = 'Imagem;'
cabecalho_mse += 'MSE Ruído Gaussiano;'
cabecalho_mse += 'MSE Ruído Spekle;'
cabecalho_mse += 'MSE Ruído Sal e Pimenta;\n'

arquivo_mse_filtro_gaussiano = codecs. open('./metricas/mse_filtro_gaussiano.txt', 'w', 'utf-8')

arquivo_mse_filtro_gaussiano.write(cabecalho_mse)

arquivo_mse_filtro_mediana = codecs. open('./metricas/mse_filtro_mediana.txt', 'w', 'utf-8')
arquivo_mse_filtro_mediana.write(cabecalho_mse)

arquivo_mse_filtro_nlm = codecs. open('./metricas/mse_filtro_nlm.txt', 'w', 'utf-8')
arquivo_mse_filtro_nlm.write(cabecalho_mse)

arquivo_mse_filtro_wiener = codecs. open('./metricas/mse_filtro_wiener.txt', 'w', 'utf-8')
arquivo_mse_filtro_wiener.write(cabecalho_mse)


lista_imagens_originais = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens_originais:
    imagem_original = img_as_float(imread(dir_imagens_originais + '/' + nome_imagem, as_gray=True))
    imagem_ruido_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle = img_as_float(imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True))
    imagem_ruido_nao_estacionario = img_as_float(imread(dir_imagens_ruido_nao_estacionario + '/' + nome_imagem, as_gray=True))

    #MSE FILTRO GAUSSIANO
    imagem_ruido_gaussiano_filtro_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano_filtro_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta_filtro_gaussiano = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta_filtro_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle_filtro_gaussiano = img_as_float(imread(dir_imagens_ruido_spekle_filtro_gaussiano + '/' + nome_imagem, as_gray=True))

    valor_mse_ruido_gaussiano_filtro_gaussiano = calcular_mse(imagem_ruido_gaussiano, imagem_ruido_gaussiano_filtro_gaussiano)
    valor_mse_ruido_spekle_filtro_gaussiano = calcular_mse(imagem_ruido_spekle, imagem_ruido_spekle_filtro_gaussiano)
    valor_mse_ruido_sal_e_pimenta_filtro_gaussiano = calcular_mse(imagem_ruido_sal_e_pimenta, imagem_ruido_sal_e_pimenta_filtro_gaussiano)

    linha = nome_imagem + ';'
    linha += str(valor_mse_ruido_gaussiano_filtro_gaussiano) + ';'
    linha += str(valor_mse_ruido_spekle_filtro_gaussiano) + ';'
    linha += str(valor_mse_ruido_sal_e_pimenta_filtro_gaussiano) + ';'
    linha += '\n'
    arquivo_mse_filtro_gaussiano.write(linha)


    # MSE FILTRO MEDIANA
    imagem_ruido_gaussiano_filtro_mediana = img_as_float(imread(dir_imagens_ruido_gaussiano_filtro_mediana + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle_filtro_mediana = img_as_float(imread(dir_imagens_ruido_spekle_filtro_mediana + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta_filtro_mediana = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta_filtro_mediana + '/' + nome_imagem, as_gray=True))

    valor_mse_ruido_gaussiano_filtro_mediana = calcular_mse(imagem_ruido_gaussiano, imagem_ruido_gaussiano_filtro_mediana)
    valor_mse_ruido_spekle_filtro_mediana = calcular_mse(imagem_ruido_spekle, imagem_ruido_spekle_filtro_mediana)
    valor_mse_ruido_sal_e_pimenta_filtro_mediana = calcular_mse(imagem_ruido_sal_e_pimenta, imagem_ruido_sal_e_pimenta_filtro_mediana)

    linha = nome_imagem + ';'
    linha += str(valor_mse_ruido_gaussiano_filtro_mediana) + ';'
    linha += str(valor_mse_ruido_spekle_filtro_mediana) + ';'
    linha += str(valor_mse_ruido_sal_e_pimenta_filtro_mediana) + ';'
    linha += '\n'
    arquivo_mse_filtro_mediana.write(linha)

    # MSE FILTRO NLM
    imagem_ruido_gaussiano_filtro_nlm = img_as_float(imread(dir_imagens_ruido_gaussiano_filtro_nlm + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle_filtro_nlm = img_as_float(imread(dir_imagens_ruido_spekle_filtro_nlm + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta_filtro_nlm = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta_filtro_nlm + '/' + nome_imagem, as_gray=True))

    valor_mse_ruido_gaussiano_filtro_nlm = calcular_mse(imagem_ruido_gaussiano, imagem_ruido_gaussiano_filtro_nlm)
    valor_mse_ruido_spekle_filtro_nlm = calcular_mse(imagem_ruido_spekle, imagem_ruido_spekle_filtro_nlm)
    valor_mse_ruido_sal_e_pimenta_filtro_nlm = calcular_mse(imagem_ruido_sal_e_pimenta, imagem_ruido_sal_e_pimenta_filtro_nlm)

    linha = nome_imagem + ';'
    linha += str(valor_mse_ruido_gaussiano_filtro_nlm) + ';'
    linha += str(valor_mse_ruido_spekle_filtro_nlm) + ';'
    linha += str(valor_mse_ruido_sal_e_pimenta_filtro_nlm) + ';'
    linha += '\n'
    arquivo_mse_filtro_nlm.write(linha)

    # MSE FILTRO WIENER
    imagem_ruido_gaussiano_filtro_wiener = img_as_float(imread(dir_imagens_ruido_gaussiano_filtro_wiener + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle_filtro_wiener = img_as_float(imread(dir_imagens_ruido_spekle_filtro_wiener + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta_filtro_wiener = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta_filtro_wiener + '/' + nome_imagem, as_gray=True))

    valor_mse_ruido_gaussiano_filtro_wiener = calcular_mse(imagem_ruido_gaussiano, imagem_ruido_gaussiano_filtro_wiener)
    valor_mse_ruido_spekle_filtro_wiener = calcular_mse(imagem_ruido_spekle, imagem_ruido_spekle_filtro_wiener)
    valor_mse_ruido_sal_e_pimenta_filtro_wiener = calcular_mse(imagem_ruido_sal_e_pimenta, imagem_ruido_sal_e_pimenta_filtro_wiener)

    linha = nome_imagem + ';'
    linha += str(valor_mse_ruido_gaussiano_filtro_wiener) + ';'
    linha += str(valor_mse_ruido_spekle_filtro_wiener) + ';'
    linha += str(valor_mse_ruido_sal_e_pimenta_filtro_wiener) + ';'
    linha += '\n'
    arquivo_mse_filtro_wiener.write(linha)




    print(nome_imagem)



arquivo_mse_filtro_gaussiano.close()
arquivo_mse_filtro_mediana.close()
arquivo_mse_filtro_nlm.close()
arquivo_mse_filtro_wiener.close()
print('FIM METRICAS MSE E SSIM')