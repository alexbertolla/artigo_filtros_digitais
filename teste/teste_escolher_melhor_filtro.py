import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from skimage.util import random_noise
from scipy import fftpack
import os
import random
from matplotlib import pylab

def calcular_psnr(img_original, img_filtrada):
    return round(psnr(img_original, img_filtrada), 2)

def calcular_mse(img_original, img_filtrada):
    return round(mse(img_original, img_filtrada), 2)

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

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano

lista_porcentagem_corte = [0.05, 0.07, 0.10, 0.13, 0.15]
lista_instensidade_ruido = [0.00, 0.05, 0.10, 0.15]
dir_imagens_ruidosas = '../imagens_ruido_gaussiano/'
dir_imagens_originais = '../banco_imagens/'

lista_psnr_filtro_5 = []
lista_psnr_filtro_7 = []
lista_psnr_filtro_10 = []
lista_psnr_filtro_13 = []
lista_psnr_filtro_15 = []

lista_mse_filtro_5 = []
lista_mse_filtro_10 = []
lista_mse_filtro_15 = []




lista_imagens = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens:
    print('Imagem: ' + nome_imagem)
    imagem_original = imread(dir_imagens_originais + nome_imagem, as_gray=True)
    #print(imagem_original.dtype)

    imagem_ruidosa = img_as_ubyte(np.zeros(imagem_original.shape))
    imagem_filtrada = img_as_ubyte(np.zeros(imagem_original.shape))

    #print(imagem_ruidosa.dtype)
    #print(imagem_filtrada.dtype)

    l, c = imagem_original.shape

    q1_original = imagem_original[:int(l/2), :int(c/2)]
    q2_original = imagem_original[:int(l/2), int(c/2):]
    q3_original = imagem_original[int(l/2):, :int(c/2)]
    q4_original = imagem_original[int(l/2):, int(c/2):]

    ruido_q1 = random.choice(lista_instensidade_ruido)
    ruido_q2 = random.choice(lista_instensidade_ruido)
    ruido_q3 = random.choice(lista_instensidade_ruido)
    ruido_q4 = random.choice(lista_instensidade_ruido)
    q1_ruidoso = img_as_ubyte(add_ruido_gaussiano(q1_original, ruido_q1))
    q2_ruidoso = img_as_ubyte(add_ruido_gaussiano(q2_original, ruido_q2))
    q3_ruidoso = img_as_ubyte(add_ruido_gaussiano(q3_original, ruido_q3))
    q4_ruidoso = img_as_ubyte(add_ruido_gaussiano(q4_original, ruido_q4))

    #print(q1_ruidoso.dtype)
    #print(q2_ruidoso.dtype)
    #print(q3_ruidoso.dtype)
    #print(q4_ruidoso.dtype)

    print('Insensidade ruído Q1: ' + str(ruido_q1))
    print('Insensidade ruído Q2: ' + str(ruido_q2))
    print('Insensidade ruído Q3: ' + str(ruido_q3))
    print('Insensidade ruído Q4: ' + str(ruido_q4))

    for porcentagem_corte in lista_porcentagem_corte:
        print('Porcentagem de corte: ' + str(porcentagem_corte * 100) + '%')

        q1_filtrado = np.array(aplicar_filtro_passa_baixa(q1_ruidoso, porcentagem_corte), dtype='uint8')
        q2_filtrado = np.array(aplicar_filtro_passa_baixa(q2_ruidoso, porcentagem_corte), dtype='uint8')
        q3_filtrado = np.array(aplicar_filtro_passa_baixa(q3_ruidoso, porcentagem_corte), dtype='uint8')
        q4_filtrado = np.array(aplicar_filtro_passa_baixa(q4_ruidoso, porcentagem_corte), dtype='uint8')

        #print(q1_filtrado.dtype)
        #print(q2_filtrado.dtype)
        #print(q3_filtrado.dtype)
        #print(q4_filtrado.dtype)

        imagem_filtrada[:int(l / 2), :int(c / 2)] = q1_filtrado
        imagem_filtrada[:int(l / 2), int(c / 2):] = q2_filtrado
        imagem_filtrada[int(l / 2):, :int(c / 2)] = q3_filtrado
        imagem_filtrada[int(l / 2):, int(c / 2):] = q4_filtrado

        imagem_ruidosa[:int(l / 2), :int(c / 2)] = q1_ruidoso
        imagem_ruidosa[:int(l / 2), int(c / 2):] = q2_ruidoso
        imagem_ruidosa[int(l / 2):, :int(c / 2)] = q3_ruidoso
        imagem_ruidosa[int(l / 2):, int(c / 2):] = q4_ruidoso

        array_psnr = []
        array_psnr.append(calcular_psnr(q1_original, q1_filtrado))
        array_psnr.append(calcular_psnr(q2_original, q2_filtrado))
        array_psnr.append(calcular_psnr(q3_original, q3_filtrado))
        array_psnr.append(calcular_psnr(q4_original, q4_filtrado))
        mediana_psnr = round(np.median(array_psnr), 2)

        array_mse = []
        array_mse.append(calcular_mse(q1_original, q1_filtrado))
        array_mse.append(calcular_mse(q2_original, q2_filtrado))
        array_mse.append(calcular_mse(q3_original, q3_filtrado))
        array_mse.append(calcular_mse(q4_original, q4_filtrado))
        mediana_mse = round(np.median(array_mse), 2)

        lista_porcentagem_corte = [0.05, 0.07, 0.10, 0.13, 0.15]
        if porcentagem_corte == 0.05:
            lista_psnr_filtro_5.append(round(np.median(array_psnr), 2))
            lista_mse_filtro_5.append(round(np.median(array_mse), 2))
        elif porcentagem_corte == 0.07:
            lista_psnr_filtro_7.append(round(np.median(array_psnr), 2))
            #lista_mse_filtro_7.append(round(np.median(array_mse), 2))
        elif porcentagem_corte == 0.10:
            lista_psnr_filtro_10.append(round(np.median(array_psnr), 2))
            lista_mse_filtro_10.append(round(np.median(array_mse), 2))
        elif porcentagem_corte == 0.13:
            lista_psnr_filtro_13.append(round(np.median(array_psnr), 2))
            #lista_mse_filtro_10.append(round(np.median(array_mse), 2))
        else:
            lista_psnr_filtro_15.append(round(np.median(array_psnr), 2))
            lista_mse_filtro_15.append(round(np.median(array_mse), 2))

        print('PSNR ('+ str(porcentagem_corte) +'): ' + str(mediana_psnr))
        print('MSE: ('+ str(porcentagem_corte) +'): ' + str(mediana_mse))



        pylab.figure()
        pylab.suptitle(nome_imagem)

        pylab.subplot(1, 3, 1), pylab.title('Imagem Original')
        pylab.imshow(imagem_original, cmap='gray')

        pylab.subplot(1, 3, 2), pylab.title('Imagem Ruidosa')
        pylab.imshow(imagem_ruidosa, cmap='gray')

        pylab.subplot(1, 3, 3), pylab.title('Imagem Filtrada (' + str(porcentagem_corte*100) + '%)')
        pylab.imshow(imagem_filtrada, cmap='gray')


        #pylab.show()
        pylab.close()

    #print(imagem_ruidosa.dtype)
    #print(imagem_filtrada.dtype)

    print()

print('lista_psnr_filtro_5: ', str(lista_psnr_filtro_5))
print('lista_psnr_filtro_7: ', str(lista_psnr_filtro_7))
print('lista_psnr_filtro_10: ', str(lista_psnr_filtro_10))
print('lista_psnr_filtro_13: ', str(lista_psnr_filtro_13))
print('lista_psnr_filtro_15: ', str(lista_psnr_filtro_15))

for idx in range(len(lista_psnr_filtro_5)):
    psnr_5 = lista_psnr_filtro_5[idx]
    psnr_7 = lista_psnr_filtro_7[idx]
    psnr_10 = lista_psnr_filtro_10[idx]
    psnr_13 = lista_psnr_filtro_13[idx]
    psnr_15 = lista_psnr_filtro_15[idx]

    if psnr_5 > psnr_7:
        if psnr_5 > psnr_10:
            if psnr_5 > psnr_13:
                if psnr_5 > psnr_15:
                    print('psnr 5% é a maior.')

    if psnr_7 > psnr_10:
        if psnr_7 > psnr_13:
            if psnr_7 > psnr_15:
                print('psnr 7% é a maior.')

    if psnr_10 > psnr_13:
        if psnr_10 > psnr_15:
            print('psnr 10% é a maior.')

    if psnr_13 > psnr_15:
        print('psnr 13% é a maior.')
    else:
        print('psnr 15% é a maior.')






exit('EXIT')
for porcentagem_corte in lista_corte:
    print('Porcentagem de corte: ' + str(porcentagem_corte*100) + '%')
    for nome_imagem in lista_imagens:
        #print(nome_imagem)
        imagem_ruidosa = img_as_float(imread(dir_imagens_ruidosas + nome_imagem, as_gray=True))
        linha, coluna = imagem_ruidosa.shape

        q1_ruidoso = imagem_ruidosa[:int(linha / 2), :int(coluna / 2)]
        q2_ruidoso = imagem_ruidosa[:int(linha / 2), int(coluna / 2):]
        q3_ruidoso = imagem_ruidosa[int(linha / 2), :int(coluna / 2)]
        q4_ruidoso = imagem_ruidosa[int(linha / 2), int(coluna / 2):]

        imagem_filtrada_passa_baixa = np.zeros(imagem_ruidosa.shape)

        imagem_filtrada_passa_baixa[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[:int(linha / 2), :int(coluna / 2)], porcentagem_corte)

        imagem_filtrada_passa_baixa[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[:int(linha / 2), int(coluna / 2):], porcentagem_corte)

        imagem_filtrada_passa_baixa[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[int(linha / 2):, :int(coluna / 2)], porcentagem_corte)

        imagem_filtrada_passa_baixa[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(
            imagem_ruidosa[int(linha / 2):, int(coluna / 2):], porcentagem_corte)

        q1_filtrado = imagem_filtrada_passa_baixa[:int(linha / 2), :int(coluna / 2)]
        q2_filtrado = imagem_filtrada_passa_baixa[:int(linha / 2), int(coluna / 2):]
        q3_filtrado = imagem_filtrada_passa_baixa[int(linha / 2), :int(coluna / 2)]
        q4_filtrado = imagem_filtrada_passa_baixa[int(linha / 2), int(coluna / 2):]

        array_psnr = []
        array_psnr.append(calcular_psnr(q1_ruidoso, q1_filtrado))
        array_psnr.append(calcular_psnr(q2_ruidoso, q2_filtrado))
        array_psnr.append(calcular_psnr(q3_ruidoso, q3_filtrado))
        array_psnr.append(calcular_psnr(q4_ruidoso, q4_filtrado))

        array_mse = []
        array_mse.append(calcular_mse(q1_ruidoso, q1_filtrado))
        array_mse.append(calcular_mse(q2_ruidoso, q2_filtrado))
        array_mse.append(calcular_mse(q3_ruidoso, q3_filtrado))
        array_mse.append(calcular_mse(q4_ruidoso, q4_filtrado))

        if porcentagem_corte == 0.05:
            lista_psnr_filtro_5.append(round(np.median(array_psnr), 2))
            lista_mse_filtro_5.append(round(np.median(array_mse), 2))
        elif porcentagem_corte == 0.10:
            lista_psnr_filtro_10.append(round(np.median(array_psnr), 2))
            lista_mse_filtro_10.append(round(np.median(array_mse), 2))
        else:
            lista_psnr_filtro_15.append(round(np.median(array_psnr), 2))
            lista_mse_filtro_15.append(round(np.median(array_mse), 2))



print('lista_psnr_filtro_5: ', str(lista_psnr_filtro_5))
print('lista_psnr_filtro_10: ', str(lista_psnr_filtro_10))
print('lista_psnr_filtro_15: ', str(lista_psnr_filtro_15))


for idx in range(len(lista_psnr_filtro_5)):
    psnr_5 = lista_psnr_filtro_5[idx]
    psnr_10 = lista_psnr_filtro_10[idx]
    psnr_15 = lista_psnr_filtro_15[idx]

    mse_5 = lista_mse_filtro_5[idx]
    mse_10 = lista_mse_filtro_10[idx]
    mse_15 = lista_mse_filtro_15[idx]

    if psnr_5 > psnr_10:
        if (psnr_5 > psnr_15) and (mse_5 < mse_15):
            print('psnr 5% é a maior e mse 5% é a menor.')
    elif (psnr_10 > psnr_15) and (mse_10 < mse_15):
            print('psnr 10% é a maior e mse 10% é a menor..')
    else:
        print('psnr 15% é a maior e mse 15% é a menor..')

    print('(' + str(psnr_5) + ', ' + str(psnr_10) + ', ' + str(psnr_15) + ')')
    print('(' + str(mse_5) + ', ' + str(mse_10) + ', ' + str(mse_15) + ')')
    print()


print('FIM ESCOLHER MELHOR FILTRO')