from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from sporco.metric import snr
from matplotlib import pylab
import os

import codecs

def calcular_snr(imagem_original, imagem_ruidosa):
    return abs(snr(imagem_original, imagem_ruidosa))

def calcular_mse(imagem_original, imagem_filtrada):
    return mean_squared_error(imagem_original, imagem_filtrada)

def calcular_ssim(imagem_original, imagem_filtrada):
    return structural_similarity(imagem_original, imagem_filtrada, multichannel=False)


dir_imagens_originais = 'imagens_originais'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'
dir_imagens_ruido_nao_estacionario = 'imagens_ruido_nao_estacionario'

array_snr = []
array_snr_gaussiano = []
array_snr_spekle = []
array_snr_sal_e_pimenta = []
array_snr_nao_estacionario = []

array_imagem = []


lista_imagens_originais = os.listdir(dir_imagens_originais)

print('IMAGEM                     SNR G                     SNR SPK                     SNR SP                     SNR NS')
pylab.figure()

arquivo = codecs. open('metricas_srn.txt', 'w', 'utf-8')
cabecalho = 'Imagem;SNR Ruído Gaussiano;SNR Ruído Spekle;SNR Ruído Sal e Pimenta;SNR Ruído Não Estacionário\n'
arquivo.write(cabecalho)

for nome_imagem in lista_imagens_originais:

    imagem_original = img_as_float(imread(dir_imagens_originais +'/' + nome_imagem, as_gray=True))
    imagem_ruido_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle = img_as_float(imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True))
    imagem_ruido_nao_estacionario = img_as_float(imread(dir_imagens_ruido_nao_estacionario + '/' + nome_imagem, as_gray=True))

    valor_snr_gaussinao = calcular_snr(imagem_original, imagem_ruido_gaussiano)
    valor_snr_spekle = calcular_snr(imagem_original, imagem_ruido_spekle)
    valor_snr_sal_e_pimenta = calcular_snr(imagem_original, imagem_ruido_sal_e_pimenta)
    valor_snr_nao_estacionario = calcular_snr(imagem_original, imagem_ruido_nao_estacionario)

    linha = nome_imagem + ';'
    linha += str(valor_snr_gaussinao) + ';'
    linha += str(valor_snr_spekle) + ';'
    linha += str(valor_snr_sal_e_pimenta) + ';'
    linha += str(valor_snr_nao_estacionario)
    linha += '\n'
    arquivo.write(linha)

    array_imagem.append('SNR G')
    array_imagem.append('SNR SPK')
    array_imagem.append('SNR SP')
    array_imagem.append('SNR NEst')

    array_snr_gaussiano.append(valor_snr_gaussinao)
    array_snr_spekle.append(valor_snr_spekle)
    array_snr_sal_e_pimenta.append(valor_snr_sal_e_pimenta)
    array_snr_nao_estacionario.append(valor_snr_nao_estacionario)

    array_snr.append(valor_snr_gaussinao)
    array_snr.append(valor_snr_spekle)
    array_snr.append(valor_snr_sal_e_pimenta)
    array_snr.append(valor_snr_nao_estacionario)

    print(nome_imagem + '       ' + str(valor_snr_gaussinao) + '       ' + str(valor_snr_spekle) + '       ' + str(
        valor_snr_sal_e_pimenta) + '       ' + str(valor_snr_nao_estacionario))


    pylab.suptitle(nome_imagem)
    pylab.bar(array_imagem, array_snr)

#pylab.show()


arquivo.close()

snr_gaissoano_max = max(array_snr_gaussiano)
snr_spekle_max = max(array_snr_spekle)
snr_sal_e_pimenta_max = max(array_snr_sal_e_pimenta)
snr_nao_estacionario_max = max(array_snr_nao_estacionario)

snr_gaissoano_min = min(array_snr_gaussiano)
snr_spekle_min = min(array_snr_spekle)
snr_sal_e_pimenta_min = min(array_snr_sal_e_pimenta)
snr_nao_estacionario_min = min(array_snr_nao_estacionario)

print('SNR Gaussiano Mín = ', snr_gaissoano_min)
#print('SNR Gaussiano Máx = ', snr_gaissoano_max)
#print()

print('SNR Spekle Mín = ', snr_spekle_min)
#print('SNR Spekle Máx = ', snr_spekle_max)
#print()

print('SNR Sal e Pimenta Min = ', snr_sal_e_pimenta_min)
#print('SNR Sal e Pimenta Máx = ', snr_sal_e_pimenta_max)
#print()

print('SNR Não Estacionário Min = ', snr_nao_estacionario_min)
#print('SNR Não Estacionário Máx = ', snr_nao_estacionario_max)


print('FIM SNR')