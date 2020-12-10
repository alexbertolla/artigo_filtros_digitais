import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from scipy import fftpack
from matplotlib import pylab
from skimage.filters import gaussian
import os
import shutil

def calcular_mse(img_ruidosa, img_filtrada):
    return round(mse(img_ruidosa, img_filtrada), 2)

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, mode='gaussian', var=sigma)
    return imagem_ruido_gaussiano

def gerar_spectro(transformata_imagem):
    shift_frq = fftpack.fftshift(transformata_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

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


def filtro_passa_alta(imagem, porcentagem_corte):
    freq = fp.fft2(imagem)
    sfreq = fp.fftshift(freq)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)
    sfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0

    imagem_passa_alta = fp.ifft2(fp.ifftshift(sfreq)).real
    espectro_filtro = (20 * np.log10(0.1 + sfreq)).real
    # , espectro_filtro
    return imagem_passa_alta


lista_porcentagem_corte = ([0.05, 0.07, 0.10, 0.13, 0.15])

caminho_imagem_original = '../banco_imagens/hz_1235134-PPT.jpg'
imagem_original = img_as_float(imread(caminho_imagem_original, as_gray=True))
linha, coluna = imagem_original.shape

quadrante_1 = imagem_original[:int(linha / 2), :int(coluna / 2)]
quadrante_2 = imagem_original[:int(linha / 2), int(coluna / 2):]
quadrante_3 = imagem_original[int(linha / 2):, int(coluna / 2):]
quadrante_4 = imagem_original[int(linha / 2):, :int(coluna / 2)]


quadrante_ruido_1 = add_ruido_gaussiano(quadrante_1, 0.00)
quadrante_ruido_2 = add_ruido_gaussiano(quadrante_2, 0.05)
quadrante_ruido_3 = add_ruido_gaussiano(quadrante_3, 0.10)
quadrante_ruido_4 = add_ruido_gaussiano(quadrante_4, 0.15)

imagem_ruido_gaussiano = np.zeros(imagem_original.shape)
imagem_ruido_gaussiano[:int(linha / 2), :int(coluna / 2)] = quadrante_ruido_1
imagem_ruido_gaussiano[:int(linha / 2), int(coluna / 2):] = quadrante_ruido_2
imagem_ruido_gaussiano[int(linha / 2):, int(coluna / 2):] = quadrante_ruido_3
imagem_ruido_gaussiano[int(linha / 2):, :int(coluna / 2)] = quadrante_ruido_4

imagem_pb_5 = np.zeros(imagem_original.shape)
imagem_pb_7 = np.zeros(imagem_original.shape)
imagem_pb_10 = np.zeros(imagem_original.shape)
imagem_pb_13 = np.zeros(imagem_original.shape)
imagem_pb_15 = np.zeros(imagem_original.shape)

imagem_pb_5[:int(linha / 2), :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_1, 0.05)
imagem_pb_5[:int(linha / 2), int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_2, 0.05)
imagem_pb_5[int(linha / 2):, int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_3, 0.05)
imagem_pb_5[int(linha / 2):, :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_4, 0.05)

imagem_pb_7[:int(linha / 2), :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_1, 0.07)
imagem_pb_7[:int(linha / 2), int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_2, 0.07)
imagem_pb_7[int(linha / 2):, int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_3, 0.07)
imagem_pb_7[int(linha / 2):, :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_4, 0.07)

imagem_pb_10[:int(linha / 2), :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_1, 0.10)
imagem_pb_10[:int(linha / 2), int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_2, 0.10)
imagem_pb_10[int(linha / 2):, int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_3, 0.10)
imagem_pb_10[int(linha / 2):, :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_4, 0.10)

imagem_pb_13[:int(linha / 2), :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_1, 0.13)
imagem_pb_13[:int(linha / 2), int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_2, 0.13)
imagem_pb_13[int(linha / 2):, int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_3, 0.13)
imagem_pb_13[int(linha / 2):, :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_4, 0.13)

imagem_pb_15[:int(linha / 2), :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_1, 0.15)
imagem_pb_15[:int(linha / 2), int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_2, 0.15)
imagem_pb_15[int(linha / 2):, int(coluna / 2):] = filtro_passa_baixa(quadrante_ruido_3, 0.15)
imagem_pb_15[int(linha / 2):, :int(coluna / 2)] = filtro_passa_baixa(quadrante_ruido_4, 0.15)

imagem_pb_5 = img_as_ubyte(imagem_pb_5)
imagem_pb_7 = img_as_ubyte(imagem_pb_7)
imagem_pb_10 = img_as_ubyte(imagem_pb_10)
imagem_pb_13 = img_as_ubyte(imagem_pb_13)
imagem_pb_15 = img_as_ubyte(imagem_pb_15)


imagem_pa_5 = np.zeros(imagem_original.shape)
imagem_pa_7 = np.zeros(imagem_original.shape)
imagem_pa_10 = np.zeros(imagem_original.shape)
imagem_pa_13 = np.zeros(imagem_original.shape)
imagem_pa_15 = np.zeros(imagem_original.shape)

imagem_pa_5[:int(linha / 2), :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_1, 0.005)
imagem_pa_5[:int(linha / 2), int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_2, 0.005)
imagem_pa_5[int(linha / 2):, int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_3, 0.005)
imagem_pa_5[int(linha / 2):, :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_4, 0.005)

imagem_pa_7[:int(linha / 2), :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_1, 0.01)
imagem_pa_7[:int(linha / 2), int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_2, 0.01)
imagem_pa_7[int(linha / 2):, int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_3, 0.01)
imagem_pa_7[int(linha / 2):, :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_4, 0.01)

imagem_pa_10[:int(linha / 2), :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_1, 0.015)
imagem_pa_10[:int(linha / 2), int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_2, 0.015)
imagem_pa_10[int(linha / 2):, int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_3, 0.015)
imagem_pa_10[int(linha / 2):, :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_4, 0.015)

imagem_pa_13[:int(linha / 2), :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_1, 0.02)
imagem_pa_13[:int(linha / 2), int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_2, 0.02)
imagem_pa_13[int(linha / 2):, int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_3, 0.02)
imagem_pa_13[int(linha / 2):, :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_4, 0.02)

imagem_pa_15[:int(linha / 2), :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_1, 0.025)
imagem_pa_15[:int(linha / 2), int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_2, 0.025)
imagem_pa_15[int(linha / 2):, int(coluna / 2):] = filtro_passa_alta(quadrante_ruido_4, 0.025)
imagem_pa_15[int(linha / 2):, :int(coluna / 2)] = filtro_passa_alta(quadrante_ruido_3, 0.025)

imagem_pa_5 = img_as_ubyte(imagem_pa_5)
imagem_pa_7 = img_as_ubyte(imagem_pa_7)
imagem_pa_10 = img_as_ubyte(imagem_pa_10)
imagem_pa_13 = img_as_ubyte(imagem_pa_13)
imagem_pa_15 = img_as_ubyte(imagem_pa_15)


imagem_final_5 = imagem_pa_5 + imagem_pb_5
imagem_final_7 = imagem_pa_7 + imagem_pb_7
imagem_final_10 = imagem_pa_10 + imagem_pb_10
imagem_final_13 = imagem_pa_13 + imagem_pb_13
imagem_final_15 = imagem_pa_15 + imagem_pb_15
print(imagem_final_5.dtype)


pylab.figure()
pylab.subplot(3, 7, 1)
pylab.axis('off')
pylab.title('Imagem Original')
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(3, 7, 2)
pylab.axis('off')
pylab.title('Imagem Ruídosa')
pylab.imshow(imagem_ruido_gaussiano, cmap='gray')

pylab.subplot(3, 7, 3)
pylab.axis('off')
pylab.title('Imagem Filtrada 5%')
pylab.imshow(imagem_pb_5, cmap='gray')

pylab.subplot(3, 7, 4)
pylab.axis('off')
pylab.title('Imagem Filtrada 7%')
pylab.imshow(imagem_pb_7, cmap='gray')

pylab.subplot(3, 7, 5)
pylab.axis('off')
pylab.title('Imagem Filtrada 10%')
pylab.imshow(imagem_pb_10, cmap='gray')

pylab.subplot(3, 7, 6)
pylab.axis('off')
pylab.title('Imagem Filtrada 13%')
pylab.imshow(imagem_pb_13, cmap='gray')

pylab.subplot(3, 7, 7)
pylab.axis('off')
pylab.title('Imagem Filtrada 15%')
pylab.imshow(imagem_pb_15, cmap='gray')

pylab.subplot(3, 7, 8)
pylab.axis('off')
pylab.title('Imagem Original')
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(3, 7, 9)
pylab.axis('off')
pylab.title('Imagem Ruídosa')
pylab.imshow(imagem_ruido_gaussiano, cmap='gray')

pylab.subplot(3, 7, 10)
pylab.axis('off')
pylab.title('Imagem Filtrada 5%')
pylab.imshow(imagem_pa_5, cmap='gray')

pylab.subplot(3, 7, 11)
pylab.axis('off')
pylab.title('Imagem Filtrada 7%')
pylab.imshow(imagem_pa_7, cmap='gray')

pylab.subplot(3, 7, 12)
pylab.axis('off')
pylab.title('Imagem Filtrada 10%')
pylab.imshow(imagem_pa_10, cmap='gray')

pylab.subplot(3, 7, 13)
pylab.axis('off')
pylab.title('Imagem Filtrada 13%')
pylab.imshow(imagem_pa_13, cmap='gray')

pylab.subplot(3, 7, 14)
pylab.axis('off')
pylab.title('Imagem Filtrada 15%')
pylab.imshow(imagem_pa_15, cmap='gray')

pylab.subplot(3, 7, 15)
pylab.axis('off')
pylab.title('Imagem Original')
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(3, 7, 16)
pylab.axis('off')
pylab.title('Imagem Ruídosa')
pylab.imshow(imagem_ruido_gaussiano, cmap='gray')

pylab.subplot(3, 7, 17)
pylab.axis('off')
pylab.title('Imagem Filtrada 5%')
pylab.imshow(imagem_final_5, cmap='gray')

pylab.subplot(3, 7, 18)
pylab.axis('off')
pylab.title('Imagem Filtrada 7%')
pylab.imshow(imagem_final_7, cmap='gray')

pylab.subplot(3, 7, 19)
pylab.axis('off')
pylab.title('Imagem Filtrada 10%')
pylab.imshow(imagem_final_10, cmap='gray')

pylab.subplot(3, 7, 20)
pylab.axis('off')
pylab.title('Imagem Filtrada 13%')
pylab.imshow(imagem_final_13, cmap='gray')

pylab.subplot(3, 7, 21)
pylab.axis('off')
pylab.title('Imagem Filtrada 15%')
pylab.imshow(imagem_final_15, cmap='gray')



pylab.show()


print('FIM TESTE FILTRO PASSA ALATA')