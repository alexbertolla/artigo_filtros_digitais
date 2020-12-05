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
dir_imagens_ruidosas = '../imagens_ruido_gaussiano/'
dir_imagens_originais = '../banco_imagens/'

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




lista_imagens = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens:
    print('Imagem: ' + nome_imagem)
    imagem_original = imread(dir_imagens_originais + nome_imagem, as_gray=True)
    #print(imagem_original.dtype)

    imagem_ruidosa = img_as_ubyte(np.zeros(imagem_original.shape))
    imagem_filtrada_pb = img_as_ubyte(np.zeros(imagem_original.shape))
    imagem_filtrada_pa = img_as_ubyte(np.zeros(imagem_original.shape))

    #print(imagem_ruidosa.dtype)
    #print(imagem_filtrada.dtype)

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

    #print(q1_ruidoso.dtype)
    #print(q2_ruidoso.dtype)
    #print(q3_ruidoso.dtype)
    #print(q4_ruidoso.dtype)

    print('Insensidade ruído Q1: ' + str(ruido_q1))
    print('Insensidade ruído Q2: ' + str(ruido_q2))
    print('Insensidade ruído Q3: ' + str(ruido_q3))
    print('Insensidade ruído Q4: ' + str(ruido_q4))

    pylab.figure()
    pylab.suptitle(nome_imagem)
    lin = 2
    col = 7
    pos = 8

    pylab.subplot(lin, col, 1), pylab.title('Imagem Original')
    pylab.axis('off')
    pylab.imshow(imagem_original, cmap='gray')

    pylab.subplot(lin, col, 2), pylab.title('Imagem Ruidosa')
    pylab.axis('off')
    pylab.imshow(imagem_ruidosa, cmap='gray')

    pylab.subplot(lin, col, 3), pylab.title('Q1 (' + str(ruido_q1) + ')')
    pylab.axis('off')
    pylab.imshow(q1_ruidoso, cmap='gray')

    pylab.subplot(lin, col, 4), pylab.title('Q2 (' + str(ruido_q2) + ')')
    pylab.axis('off')
    pylab.imshow(q2_ruidoso, cmap='gray')

    pylab.subplot(lin, col, 5), pylab.title('Q3 (' + str(ruido_q3) + ')')
    pylab.axis('off')
    pylab.imshow(q3_ruidoso, cmap='gray')

    pylab.subplot(lin, col, 6), pylab.title('Q4 (' + str(ruido_q4) + ')')
    pylab.axis('off')
    pylab.imshow(q4_ruidoso, cmap='gray')

    for porcentagem_corte in lista_porcentagem_corte:
        print('Porcentagem de corte: ' + str(porcentagem_corte * 100) + '%')

        q1_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q1_ruidoso, porcentagem_corte), dtype='uint8')
        q2_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q2_ruidoso, porcentagem_corte), dtype='uint8')
        q3_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q3_ruidoso, porcentagem_corte), dtype='uint8')
        q4_filtrado_pb = np.array(aplicar_filtro_passa_baixa(q4_ruidoso, porcentagem_corte), dtype='uint8')

        q1_filtrado_pa = np.array(aplicar_filtro_passa_alta(q1_ruidoso, porcentagem_corte), dtype='uint8')
        q2_filtrado_pa = np.array(aplicar_filtro_passa_alta(q2_ruidoso, porcentagem_corte), dtype='uint8')
        q3_filtrado_pa = np.array(aplicar_filtro_passa_alta(q3_ruidoso, porcentagem_corte), dtype='uint8')
        q4_filtrado_pa = np.array(aplicar_filtro_passa_alta(q4_ruidoso, porcentagem_corte), dtype='uint8')

        #print(q1_filtrado.dtype)
        #print(q2_filtrado.dtype)
        #print(q3_filtrado.dtype)
        #print(q4_filtrado.dtype)

        imagem_filtrada_pb[:int(l / 2), :int(c / 2)] = q1_filtrado_pb
        imagem_filtrada_pb[:int(l / 2), int(c / 2):] = q2_filtrado_pb
        imagem_filtrada_pb[int(l / 2):, :int(c / 2)] = q3_filtrado_pb
        imagem_filtrada_pb[int(l / 2):, int(c / 2):] = q4_filtrado_pb

        imagem_filtrada_pa[:int(l / 2), :int(c / 2)] = q1_filtrado_pa
        imagem_filtrada_pa[:int(l / 2), int(c / 2):] = q2_filtrado_pa
        imagem_filtrada_pa[int(l / 2):, :int(c / 2)] = q3_filtrado_pa
        imagem_filtrada_pa[int(l / 2):, int(c / 2):] = q4_filtrado_pa

        #imagem_ruidosa[:int(l / 2), :int(c / 2)] = q1_ruidoso
        #imagem_ruidosa[:int(l / 2), int(c / 2):] = q2_ruidoso
        #imagem_ruidosa[int(l / 2):, :int(c / 2)] = q3_ruidoso
        #imagem_ruidosa[int(l / 2):, int(c / 2):] = q4_ruidoso

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

        #lista_porcentagem_corte = [0.05, 0.07, 0.10, 0.13, 0.15]
        if porcentagem_corte == 0.05:
            lista_psnr_filtro_5_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_5_pa.append(round(np.median(array_psnr_pa), 2))
            #lista_mse_filtro_5_pb.append(round(np.median(array_mse), 2))
        elif porcentagem_corte == 0.07:
            lista_psnr_filtro_7_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_7_pa.append(round(np.median(array_psnr_pa), 2))
            #lista_mse_filtro_7.append(round(np.median(array_mse), 2))
        elif porcentagem_corte == 0.10:
            lista_psnr_filtro_10_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_10_pa.append(round(np.median(array_psnr_pa), 2))
        elif porcentagem_corte == 0.13:
            lista_psnr_filtro_13_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_13_pa.append(round(np.median(array_psnr_pa), 2))
        else:
            lista_psnr_filtro_15_pb.append(round(np.median(array_psnr_pb), 2))
            lista_psnr_filtro_15_pa.append(round(np.median(array_psnr_pa), 2))

        print('PSNR FILTRO PASSA BAIXA ('+ str(porcentagem_corte) +'): ' + str(mediana_psnr_pb))
        print('PSNR FILTRO PASSA ALTA ('+ str(porcentagem_corte) +'): ' + str(mediana_psnr_pa))
        #print()



        pylab.subplot(lin, col, pos), pylab.title('Imagem Filtrada (' + str(round(porcentagem_corte*100, 2)) + '%)')
        pylab.axis('off')
        pylab.imshow(imagem_filtrada_pb, cmap='gray')
        pos += 1


    #pylab.show()
    pylab.close()

    #print(imagem_ruidosa.dtype)
    #print(imagem_filtrada.dtype)

    print()

#print('lista_psnr_filtro_5: ', str(lista_psnr_filtro_5))
#print('lista_psnr_filtro_7: ', str(lista_psnr_filtro_7))
#print('lista_psnr_filtro_10: ', str(lista_psnr_filtro_10))
#print('lista_psnr_filtro_13: ', str(lista_psnr_filtro_13))
#print('lista_psnr_filtro_15: ', str(lista_psnr_filtro_15))

#print('Filtro 5%: Menor Valor: ' + str(np.min(lista_psnr_filtro_5)) + '. Maior Valor: ' + str(np.max(lista_psnr_filtro_5)))
#print('Filtro 7%: Menor Valor: ' + str(np.min(lista_psnr_filtro_7)) + '. Maior Valor: ' + str(np.max(lista_psnr_filtro_7)))
#print('Filtro 10%: Menor Valor: ' + str(np.min(lista_psnr_filtro_10)) + '. Maior Valor: ' + str(np.max(lista_psnr_filtro_10)))
#print('Filtro 13%: Menor Valor: ' + str(np.min(lista_psnr_filtro_13)) + '. Maior Valor: ' + str(np.max(lista_psnr_filtro_13)))
#print('Filtro 15%: Menor Valor: ' + str(np.min(lista_psnr_filtro_15)) + '. Maior Valor: ' + str(np.max(lista_psnr_filtro_15)))

idx_pb = 0
idx_pa = 0

for idx in range(len(lista_psnr_filtro_5_pb)):
    psnr_5_pb = lista_psnr_filtro_5_pb[idx]
    psnr_7_pb = lista_psnr_filtro_7_pb[idx]
    psnr_10_pb = lista_psnr_filtro_10_pb[idx]
    psnr_13_pb = lista_psnr_filtro_13_pb[idx]
    psnr_15_pb = lista_psnr_filtro_15_pb[idx]

    psnr_5_pa = lista_psnr_filtro_5_pa[idx]
    psnr_7_pa = lista_psnr_filtro_7_pa[idx]
    psnr_10_pa = lista_psnr_filtro_10_pa[idx]
    psnr_13_pa = lista_psnr_filtro_13_pa[idx]
    psnr_15_pa = lista_psnr_filtro_15_pa[idx]

    if psnr_5_pb > psnr_7_pb:
        if psnr_5_pb > psnr_10_pb:
            if psnr_5_pb > psnr_13_pb:
                if psnr_5_pb > psnr_15_pb:
                    idx_pb = idx
                    print('psnr pb 5% é a maior.')
    if psnr_5_pa > psnr_7_pa:
        if psnr_5_pa > psnr_10_pa:
            if psnr_5_pa > psnr_13_pa:
                if psnr_5_pa > psnr_15_pa:
                    idx_pb = idx
                    print('psnr pa 5% é a maior.')

    if psnr_7_pa > psnr_10_pa:
        if psnr_7_pa > psnr_13_pa:
            if psnr_7_pa > psnr_15_pa:
                idx_pb = idx
                print('psnr pa 7% é a maior.')

    if psnr_10_pa > psnr_13_pa:
        if psnr_10_pa > psnr_15_pa:
            print('psnr pa 10% é a maior.')

    if psnr_13_pa > psnr_15_pa:
        print('psnr pa 13% é a maior.')
    else:
        print('psnr pa 15% é a maior.')

print('Total de Imagens: ' + str(len(lista_imagens)))
print('Total PSNR FPB 5%:' + str(len(lista_psnr_filtro_5_pb)))
print('Total PSNR FPA 5%:' + str(len(lista_psnr_filtro_5_pa)))



print('FIM ESCOLHER MELHOR FILTRO')