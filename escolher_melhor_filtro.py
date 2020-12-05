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

def calcular_psnr(img_original, img_filtrada):
    return round(psnr(img_original, img_filtrada), 2)

def calcular_mse(img_original, img_filtrada):
    return round(mse(img_original, img_filtrada), 2)

def gerar_spectro(imagem):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

def aplicar_filtro_passa_baixa(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    shift_frq_low = np.copy(shift_frq)

    shift_frq_low[half_w - (int(half_w * porcentagem_corte)):half_w + (int(half_w * porcentagem_corte)) + 1,
    half_h - (int(half_h * porcentagem_corte)):half_h + (int(half_h * porcentagem_corte)) + 1] = 0

    shift_frq -= shift_frq_low
    imagem_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq)).real
    return imagem_filtrada

def aplicar_filtro_passa_alta(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    shift_frq[half_w - (int(half_w * porcentagem_corte)):half_w + (int(half_w * porcentagem_corte)) + 1, half_h - (int(half_h * porcentagem_corte)):half_h + (int(half_h * porcentagem_corte)) + 1] = 0
    imagem_filtrada = np.clip(fp.ifft2(fftpack.ifftshift(shift_frq)).real, 0, 255)
    return imagem_filtrada

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano

lista_porcentagem_corte = [0.05, 0.07, 0.10, 0.13, 0.15]
lista_instensidade_ruido = [0.00, 0.05, 0.10, 0.15]
dir_imagens_originais = '../banco_imagens/'
dir_imagens_filtro_passa_banda = '../imagens_filtro_passa_banda/'
shutil.rmtree(dir_imagens_filtro_passa_banda, ignore_errors=True)
os.mkdir(dir_imagens_filtro_passa_banda)

lista_psnr_filtro_5_pb = []
lista_psnr_filtro_7_pb = []
lista_psnr_filtro_10_pb = []
lista_psnr_filtro_13_pb = []
lista_psnr_filtro_15_pb = []

lista_psnr_filtro_5_pa = []
lista_psnr_filtro_7_pa = []
lista_psnr_filtro_10_pa = []
lista_psnr_filtro_13_pa = []
lista_psnr_filtro_15_pa = []

lista_mse_filtro_5_pb = []
lista_mse_filtro_7_pb = []
lista_mse_filtro_10_pb = []
lista_mse_filtro_13_pb = []
lista_mse_filtro_15_pb = []

lista_mse_filtro_5_pa = []
lista_mse_filtro_7_pa = []
lista_mse_filtro_10_pa = []
lista_mse_filtro_13_pa = []
lista_mse_filtro_15_pa = []

lista_imagem_filtrada_5_pb = []
lista_imagem_filtrada_7_pb = []
lista_imagem_filtrada_10_pb = []
lista_imagem_filtrada_13_pb = []
lista_imagem_filtrada_15_pb = []

lista_imagem_filtrada_5_pa = []
lista_imagem_filtrada_7_pa = []
lista_imagem_filtrada_10_pa = []
lista_imagem_filtrada_13_pa = []
lista_imagem_filtrada_15_pa = []

lista_imagem_filtrada_pb_final = []
lista_imagem_filtrada_pa_final = []
lista_dados_filtragem_pb = []
lista_dados_filtragem_pa = []

print('PROCESSANDO....')

lista_imagens = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens:
    imagem_original = imread(dir_imagens_originais + nome_imagem, as_gray=True)

    imagem_ruidosa = img_as_ubyte(np.zeros(imagem_original.shape))
    imagem_filtrada_pb = img_as_ubyte(np.zeros(imagem_original.shape))
    imagem_filtrada_pa = img_as_ubyte(np.zeros(imagem_original.shape))
    l, c = imagem_original.shape

    q1_original = imagem_original[:int(l/2), :int(c/2)]
    q2_original = imagem_original[:int(l/2), int(c/2):]
    q3_original = imagem_original[int(l/2):, :int(c/2)]
    q4_original = imagem_original[int(l/2):, int(c/2):]

    spectro_imagem_original = gerar_spectro(imagem_original)
    spectro_imagem_original_quadrantes = np.zeros(spectro_imagem_original.shape)
    spectro_imagem_original_quadrantes[:int(l/2), :int(c/2)] = gerar_spectro(q1_original)
    spectro_imagem_original_quadrantes[:int(l/2), int(c/2):] = gerar_spectro(q2_original)
    spectro_imagem_original_quadrantes[int(l/2):, :int(c/2)] = gerar_spectro(q3_original)
    spectro_imagem_original_quadrantes[int(l/2):, int(c/2):] = gerar_spectro(q4_original)

    ruido_q1 = random.choice(lista_instensidade_ruido)
    ruido_q2 = random.choice(lista_instensidade_ruido)
    ruido_q3 = random.choice(lista_instensidade_ruido)
    ruido_q4 = random.choice(lista_instensidade_ruido)
    q1_ruidoso = img_as_ubyte(add_ruido_gaussiano(q1_original, ruido_q1))
    q2_ruidoso = img_as_ubyte(add_ruido_gaussiano(q2_original, ruido_q2))
    q3_ruidoso = img_as_ubyte(add_ruido_gaussiano(q3_original, ruido_q3))
    q4_ruidoso = img_as_ubyte(add_ruido_gaussiano(q4_original, ruido_q4))

    imagem_ruidosa[:int(l / 2), :int(c / 2)] = q1_ruidoso
    imagem_ruidosa[:int(l / 2), int(c / 2):] = q2_ruidoso
    imagem_ruidosa[int(l / 2):, :int(c / 2)] = q3_ruidoso
    imagem_ruidosa[int(l / 2):, int(c / 2):] = q4_ruidoso


    for porcentagem_corte in lista_porcentagem_corte:
        #print('Porcentagem de corte: ' + str(porcentagem_corte * 100) + '%')

        q1_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q1_ruidoso, porcentagem_corte), dtype='uint8')
        q2_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q2_ruidoso, porcentagem_corte), dtype='uint8')
        q3_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q3_ruidoso, porcentagem_corte), dtype='uint8')
        q4_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q4_ruidoso, porcentagem_corte), dtype='uint8')

        q1_filtrado_pa = np.array(aplicar_filtro_passa_alta(q1_ruidoso, porcentagem_corte), dtype='uint8')
        q2_filtrado_pa = np.array(aplicar_filtro_passa_alta(q2_ruidoso, porcentagem_corte), dtype='uint8')
        q3_filtrado_pa = np.array(aplicar_filtro_passa_alta(q3_ruidoso, porcentagem_corte), dtype='uint8')
        q4_filtrado_pa = np.array(aplicar_filtro_passa_alta(q4_ruidoso, porcentagem_corte), dtype='uint8')


        imagem_filtrada_pb[:int(l / 2), :int(c / 2)] = q1_filtrado_pb
        imagem_filtrada_pb[:int(l / 2), int(c / 2):] = q2_filtrado_pb
        imagem_filtrada_pb[int(l / 2):, :int(c / 2)] = q3_filtrado_pb
        imagem_filtrada_pb[int(l / 2):, int(c / 2):] = q4_filtrado_pb

        imagem_filtrada_pa[:int(l / 2), :int(c / 2)] = q1_filtrado_pa
        imagem_filtrada_pa[:int(l / 2), int(c / 2):] = q2_filtrado_pa
        imagem_filtrada_pa[int(l / 2):, :int(c / 2)] = q3_filtrado_pa
        imagem_filtrada_pa[int(l / 2):, int(c / 2):] = q4_filtrado_pa


        array_psnr_pb = []
        array_psnr_pb.append(calcular_psnr(q1_original, q1_filtrado_pb))
        array_psnr_pb.append(calcular_psnr(q2_original, q2_filtrado_pb))
        array_psnr_pb.append(calcular_psnr(q3_original, q3_filtrado_pb))
        array_psnr_pb.append(calcular_psnr(q4_original, q4_filtrado_pb))
        mediana_psnr_pb = round(np.median(array_psnr_pb), 2)

        array_psnr_pa = []
        array_psnr_pa.append(calcular_psnr(q1_original, q1_filtrado_pa))
        array_psnr_pa.append(calcular_psnr(q2_original, q2_filtrado_pa))
        array_psnr_pa.append(calcular_psnr(q3_original, q3_filtrado_pa))
        array_psnr_pa.append(calcular_psnr(q4_original, q4_filtrado_pa))
        mediana_psnr_pa = round(np.median(array_psnr_pa), 2)

        array_mse = []
        array_mse.append(calcular_mse(q1_original, q1_filtrado_pb))
        array_mse.append(calcular_mse(q2_original, q2_filtrado_pb))
        array_mse.append(calcular_mse(q3_original, q3_filtrado_pb))
        array_mse.append(calcular_mse(q4_original, q4_filtrado_pb))
        mediana_mse = round(np.median(array_mse), 2)

        if porcentagem_corte == 0.05:
            lista_psnr_filtro_5_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_5_pa.append(round(np.median(array_psnr_pa), 2))

            lista_imagem_filtrada_5_pb.append(imagem_filtrada_pb)
            lista_imagem_filtrada_5_pa.append(imagem_filtrada_pa)
        elif porcentagem_corte == 0.07:
            lista_psnr_filtro_7_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_7_pa.append(round(np.median(array_psnr_pa), 2))

            lista_imagem_filtrada_7_pb.append(imagem_filtrada_pb)
            lista_imagem_filtrada_7_pa.append(imagem_filtrada_pa)

        elif porcentagem_corte == 0.10:
            lista_psnr_filtro_10_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_10_pa.append(round(np.median(array_psnr_pa), 2))

            lista_imagem_filtrada_10_pb.append(imagem_filtrada_pb)
            lista_imagem_filtrada_10_pa.append(imagem_filtrada_pa)

        elif porcentagem_corte == 0.13:
            lista_psnr_filtro_13_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_13_pa.append(round(np.median(array_psnr_pa), 2))

            lista_imagem_filtrada_13_pb.append(imagem_filtrada_pb)
            lista_imagem_filtrada_13_pa.append(imagem_filtrada_pa)

        else:
            lista_psnr_filtro_15_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_15_pa.append(round(np.median(array_psnr_pa), 2))

            lista_imagem_filtrada_15_pb.append(imagem_filtrada_pb)
            lista_imagem_filtrada_15_pa.append(imagem_filtrada_pa)

for idx_pb in range(len(lista_psnr_filtro_5_pb)):
    imagem_filtrada_pb_ideal = np.zeros(imagem_original.shape)
    psnr_5_pb = lista_psnr_filtro_5_pb[idx_pb]
    psnr_7_pb = lista_psnr_filtro_7_pb[idx_pb]
    psnr_10_pb = lista_psnr_filtro_10_pb[idx_pb]
    psnr_13_pb = lista_psnr_filtro_13_pb[idx_pb]
    psnr_15_pb = lista_psnr_filtro_15_pb[idx_pb]
    if psnr_5_pb > psnr_7_pb:
        if psnr_5_pb > psnr_10_pb:
            if psnr_5_pb > psnr_13_pb:
                if psnr_5_pb > psnr_15_pb:
                    lista_imagem_filtrada_pb_final.append(lista_imagem_filtrada_5_pb[idx_pb])
                    lista_dados_filtragem_pb.append([lista_imagens[idx_pb], '5%', psnr_5_pb])
                    continue
    if psnr_7_pb > psnr_10_pb:
        if psnr_7_pb > psnr_13_pb:
            if psnr_7_pb > psnr_15_pb:
                lista_imagem_filtrada_pb_final.append(lista_imagem_filtrada_7_pb[idx_pb])
                lista_dados_filtragem_pb.append([lista_imagens[idx_pb], '7%', psnr_7_pb])
                continue
    if psnr_10_pb > psnr_13_pb:
        if psnr_10_pb > psnr_15_pb:
            lista_imagem_filtrada_pb_final.append(lista_imagem_filtrada_10_pb[idx_pb])
            lista_dados_filtragem_pb.append([lista_imagens[idx_pb], '10%', psnr_10_pb])
            continue
    if psnr_13_pb > psnr_15_pb:
        lista_imagem_filtrada_pb_final.append(lista_imagem_filtrada_13_pb[idx_pb])
        lista_dados_filtragem_pb.append([lista_imagens[idx_pb], '13%', psnr_13_pb])
    else:
        lista_imagem_filtrada_pb_final.append(lista_imagem_filtrada_15_pb[idx_pb])
        lista_dados_filtragem_pb.append([lista_imagens[idx_pb], '15%', psnr_15_pb])


for idx_pa in range(len(lista_psnr_filtro_5_pa)):
    imagem_filtrada_pa_ideal = np.zeros(imagem_original.shape)
    psnr_5_pa = lista_psnr_filtro_5_pa[idx_pa]
    psnr_7_pa = lista_psnr_filtro_7_pa[idx_pa]
    psnr_10_pa = lista_psnr_filtro_10_pa[idx_pa]
    psnr_13_pa = lista_psnr_filtro_13_pa[idx_pa]
    psnr_15_pa = lista_psnr_filtro_15_pa[idx_pa]

    if psnr_5_pa > psnr_7_pa:
        if psnr_5_pa > psnr_10_pa:
            if psnr_5_pa > psnr_13_pa:
                if psnr_5_pa > psnr_15_pa:
                    lista_imagem_filtrada_pa_final.append(lista_imagem_filtrada_5_pa[idx_pa])
                    lista_dados_filtragem_pa.append([lista_imagens[idx_pa], '5%', psnr_5_pa])
                    continue
    if psnr_7_pa > psnr_10_pa:
        if psnr_7_pa > psnr_13_pa:
            if psnr_7_pa > psnr_15_pa:
                lista_imagem_filtrada_pa_final.append(lista_imagem_filtrada_7_pa[idx_pa])
                lista_dados_filtragem_pa.append([lista_imagens[idx_pa], '7%', psnr_7_pa])
                continue
    if psnr_10_pa > psnr_13_pa:
        if psnr_10_pa > psnr_15_pa:
            lista_imagem_filtrada_pa_final.append(lista_imagem_filtrada_10_pa[idx_pa])
            lista_dados_filtragem_pa.append([lista_imagens[idx_pa], '10%', psnr_10_pa])
            continue
    if psnr_13_pa > psnr_15_pa:
        lista_imagem_filtrada_pa_final.append(lista_imagem_filtrada_13_pa[idx_pa])
        lista_dados_filtragem_pa.append([lista_imagens[idx_pa], '13%', psnr_13_pa])
    else:
        lista_imagem_filtrada_pa_final.append(lista_imagem_filtrada_15_pa[idx_pa])
        lista_dados_filtragem_pa.append([lista_imagens[idx_pa], '15%', psnr_15_pa])
print('#######################################')
print()
print('RESULTADO FINAL')

for idx in range(len(lista_imagem_filtrada_pb_final)):
    dados_filtragem_pb = lista_dados_filtragem_pb[idx]
    dados_filtragem_pa = lista_dados_filtragem_pa[idx]

    imagem_final = lista_imagem_filtrada_pb_final[idx] + lista_imagem_filtrada_pa_final[idx]
    imsave(dir_imagens_filtro_passa_banda + dados_filtragem_pb[0], imagem_final)

    print('Imagem: ' + dados_filtragem_pb[0])
    print('Porcentagem de Corte Filtro Passa Baixa: ' + dados_filtragem_pb[1] + ', PSNR Filtro Passa Baixa: ' + str(dados_filtragem_pb[2]))
    print('Porcentagem de Corte Filtro Passa Alta: ' + dados_filtragem_pa[1] +', PSNR Filtro Passa Alta: ' + str(dados_filtragem_pa[2]))
    print(dir_imagens_filtro_passa_banda + dados_filtragem_pb[0])
    print()


print('FIM ESCOLHER MELHOR FILTRO')