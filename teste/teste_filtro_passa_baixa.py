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

def calcular_psnr(img_original, img_filtrada):
    return round(psnr(img_original, img_filtrada), 2)

def calcular_mse(img_original, img_filtrada):
    return round(mse(img_original, img_filtrada), 2)

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, mode='gaussian', var=sigma)
    return imagem_ruido_gaussiano

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
    spectro_imagem_original = gerar_spectro(discrete_transform_imagem)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    shift_frq_low = np.copy(shift_frq)
    

    shift_frq_low[half_w - (int(half_w * porcentagem_corte)):half_w + (int(half_w * porcentagem_corte)) + 1, half_h - (int(half_h * porcentagem_corte)):half_h + (int(half_h * porcentagem_corte)) + 1] = 0

    shift_frq -= shift_frq_low
    imagem_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq)).real
    return imagem_filtrada

def aplicar_filtro_passa_alta(imagem, porcentagem_corte):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    spectro_imagem_original = gerar_spectro(discrete_transform_imagem)

    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    print(half_w)
    print(int(half_w * porcentagem_corte))

    shift_frq[half_w - (int(half_w * porcentagem_corte)):half_w + (int(half_w * porcentagem_corte)) + 1, half_h - (int(half_h * porcentagem_corte)):half_h + (int(half_h * porcentagem_corte)) + 1] = 0
    imagem_filtrada = np.clip(fp.ifft2(fftpack.ifftshift(shift_frq)).real, 0, 255)

    return imagem_filtrada

def plotar(imagem, titulo, l, c, p):
    pylab.subplot(l, c, p)
    pylab.axis('off')
    pylab.title(titulo)
    pylab.imshow(imagem, cmap='gray')

lista_porcentagem_corte = ([0.05, 0.07, 0.10, 0.13, 0.15])

caminho_imagem_original = '../banco_imagens/hz_1235134-PPT.jpg'
imagem_original = imread(caminho_imagem_original, as_gray=True)
print(imagem_original.dtype)
linha, coluna = imagem_original.shape

quadrante_1 = imagem_original[:int(linha / 2), :int(coluna / 2)]
quadrante_2 = imagem_original[:int(linha / 2), int(coluna / 2):]
quadrante_3 = imagem_original[int(linha / 2):, :int(coluna / 2)]
quadrante_4 = imagem_original[int(linha / 2):, int(coluna / 2):]

spectro_imagem_original = gerar_spectro(imagem_original)
spectro_imagem_original_quadrantes = np.zeros(spectro_imagem_original.shape)
spectro_imagem_original_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(quadrante_1)
spectro_imagem_original_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(quadrante_2)
spectro_imagem_original_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(quadrante_3)
spectro_imagem_original_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(quadrante_4)


quadrante_ruido_1 = img_as_ubyte(add_ruido_gaussiano(quadrante_1, 0.00))
quadrante_ruido_2 = img_as_ubyte(add_ruido_gaussiano(quadrante_2, 0.05))
quadrante_ruido_3 = img_as_ubyte(add_ruido_gaussiano(quadrante_3, 0.10))
quadrante_ruido_4 = img_as_ubyte(add_ruido_gaussiano(quadrante_4, 0.15))

imagem_ruido_gaussiano = img_as_ubyte(np.zeros(imagem_original.shape))
imagem_ruido_gaussiano[:int(linha / 2), :int(coluna / 2)] = quadrante_ruido_1
imagem_ruido_gaussiano[:int(linha / 2), int(coluna / 2):] = quadrante_ruido_2
imagem_ruido_gaussiano[int(linha / 2):, :int(coluna / 2)] = quadrante_ruido_3
imagem_ruido_gaussiano[int(linha / 2):, int(coluna / 2):] = quadrante_ruido_4
print(imagem_ruido_gaussiano.dtype)


spectro_imagem_ruidosa = gerar_spectro(imagem_ruido_gaussiano)
spectro_ruido_quadrantes = np.zeros(spectro_imagem_ruidosa.shape)
spectro_ruido_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(quadrante_ruido_1)
spectro_ruido_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(quadrante_ruido_2)
spectro_ruido_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(quadrante_ruido_3)
spectro_ruido_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(quadrante_ruido_4)

imagem_final_5 = img_as_ubyte(np.zeros(imagem_original.shape))
imagem_final_7 = img_as_ubyte(np.zeros(imagem_original.shape))
imagem_final_10 = img_as_ubyte(np.zeros(imagem_original.shape))
imagem_final_13 = img_as_ubyte(np.zeros(imagem_original.shape))
imagem_final_15 = img_as_ubyte(np.zeros(imagem_original.shape))

imagem_final_5[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_1, 0.05)
imagem_final_5[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_2, 0.05)
imagem_final_5[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_3, 0.05)
imagem_final_5[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_4, 0.05)

imagem_final_7[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_1, 0.07)
imagem_final_7[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_2, 0.07)
imagem_final_7[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_3, 0.07)
imagem_final_7[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_4, 0.07)

imagem_final_10[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_1, 0.10)
imagem_final_10[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_2, 0.10)
imagem_final_10[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_3, 0.10)
imagem_final_10[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_4, 0.10)

imagem_final_13[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_1, 0.13)
imagem_final_13[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_2, 0.13)
imagem_final_13[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_3, 0.13)
imagem_final_13[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_4, 0.13)

imagem_final_15[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_1, 0.15)
imagem_final_15[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_2, 0.15)
imagem_final_15[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_baixa(quadrante_ruido_3, 0.15)
imagem_final_15[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_baixa(quadrante_ruido_4, 0.15)

print(imagem_final_5.dtype)
print(imagem_final_10.dtype)
print(imagem_final_15.dtype)
#imagem_final_5 = img_as_ubyte(imagem_final_5)
#imagem_final_10 = img_as_ubyte(imagem_final_10)
#imagem_final_15 = img_as_ubyte(imagem_final_15)


spectro_imagem_final_5 = gerar_spectro(imagem_final_5)
spectro_5_quadrantes = np.zeros(spectro_imagem_final_5.shape)
spectro_5_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(imagem_final_5[:int(linha / 2), :int(coluna / 2)])
spectro_5_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(imagem_final_5[:int(linha / 2), int(coluna / 2):])
spectro_5_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(imagem_final_5[int(linha / 2):, :int(coluna / 2)])
spectro_5_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(imagem_final_5[int(linha / 2):, int(coluna / 2):])

spectro_imagem_final_7 = gerar_spectro(imagem_final_7)
spectro_7_quadrantes = np.zeros(spectro_imagem_final_7.shape)
spectro_7_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(imagem_final_7[:int(linha / 2), :int(coluna / 2)])
spectro_7_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(imagem_final_7[:int(linha / 2), int(coluna / 2):])
spectro_7_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(imagem_final_7[int(linha / 2):, :int(coluna / 2)])
spectro_7_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(imagem_final_7[int(linha / 2):, int(coluna / 2):])

spectro_imagem_final_10 = gerar_spectro(imagem_final_10)
spectro_10_quadrantes = np.zeros(spectro_imagem_final_10.shape)
spectro_10_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(imagem_final_10[:int(linha / 2), :int(coluna / 2)])
spectro_10_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(imagem_final_10[:int(linha / 2), int(coluna / 2):])
spectro_10_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(imagem_final_10[int(linha / 2):, :int(coluna / 2)])
spectro_10_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(imagem_final_10[int(linha / 2):, int(coluna / 2):])

spectro_imagem_final_13 = gerar_spectro(imagem_final_13)
spectro_13_quadrantes = np.zeros(spectro_imagem_final_13.shape)
spectro_13_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(imagem_final_13[:int(linha / 2), :int(coluna / 2)])
spectro_13_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(imagem_final_13[:int(linha / 2), int(coluna / 2):])
spectro_13_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(imagem_final_13[int(linha / 2):, :int(coluna / 2)])
spectro_13_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(imagem_final_13[int(linha / 2):, int(coluna / 2):])

spectro_imagem_final_15 = gerar_spectro(imagem_final_15)
spectro_15_quadrantes = np.zeros(spectro_imagem_final_15.shape)
spectro_15_quadrantes[:int(linha / 2), :int(coluna / 2)] = gerar_spectro(imagem_final_15[:int(linha / 2), :int(coluna / 2)])
spectro_15_quadrantes[:int(linha / 2), int(coluna / 2):] = gerar_spectro(imagem_final_15[:int(linha / 2), int(coluna / 2):])
spectro_15_quadrantes[int(linha / 2):, :int(coluna / 2)] = gerar_spectro(imagem_final_15[int(linha / 2):, :int(coluna / 2)])
spectro_15_quadrantes[int(linha / 2):, int(coluna / 2):] = gerar_spectro(imagem_final_15[int(linha / 2):, int(coluna / 2):])


#print(img_as_ubyte(imagem_final_5))

psnr_q1 = calcular_psnr(quadrante_1, quadrante_ruido_1)
psnr_q2 = calcular_psnr(quadrante_2, quadrante_ruido_2)
psnr_q3 = calcular_psnr(quadrante_3, quadrante_ruido_3)
psnr_q4 = calcular_psnr(quadrante_4, quadrante_ruido_4)

mse_q1 = calcular_mse(quadrante_1, quadrante_ruido_1)
mse_q2 = calcular_mse(quadrante_2, quadrante_ruido_2)
mse_q3 = calcular_mse(quadrante_3, quadrante_ruido_3)
mse_q4 = calcular_mse(quadrante_4, quadrante_ruido_4)

print('Mediana PSNR = ' + str(np.median([psnr_q1, psnr_q2, psnr_q3, psnr_q4])))
print('Mediana MSE = ' + str(np.median([mse_q1, mse_q2, mse_q3, mse_q4])))

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
pylab.imshow(imagem_final_5, cmap='gray')

pylab.subplot(3, 7, 4)
pylab.axis('off')
pylab.title('Imagem Filtrada 7%')
pylab.imshow(imagem_final_7, cmap='gray')

pylab.subplot(3, 7, 5)
pylab.axis('off')
pylab.title('Imagem Filtrada 10%')
pylab.imshow(imagem_final_10, cmap='gray')

pylab.subplot(3, 7, 6)
pylab.axis('off')
pylab.title('Imagem Filtrada 13%')
pylab.imshow(imagem_final_13, cmap='gray')

pylab.subplot(3, 7, 7)
pylab.axis('off')
pylab.title('Imagem Filtrada 15%')
pylab.imshow(imagem_final_15, cmap='gray')

pylab.subplot(3, 7, 8)
pylab.axis('off')
pylab.title('Spectro Imagem Original')
pylab.imshow(spectro_imagem_original, cmap='gray')

pylab.subplot(3, 7, 9)
pylab.axis('off')
pylab.title('Spectro Imagem Ruidosa')
pylab.imshow(spectro_imagem_ruidosa, cmap='gray')

pylab.subplot(3, 7, 10)
pylab.axis('off')
pylab.title('Spectro 5%')
pylab.imshow(spectro_imagem_final_5, cmap='gray')

pylab.subplot(3, 7, 11)
pylab.axis('off')
pylab.title('Spectro 7%')
pylab.imshow(spectro_imagem_final_7, cmap='gray')

pylab.subplot(3, 7, 12)
pylab.axis('off')
pylab.title('Spectro 10%')
pylab.imshow(spectro_imagem_final_10, cmap='gray')

pylab.subplot(3, 7, 13)
pylab.axis('off')
pylab.title('Spectro 13%')
pylab.imshow(spectro_imagem_final_13, cmap='gray')

pylab.subplot(3, 7, 14)
pylab.axis('off')
pylab.title('Spectro 15%')
pylab.imshow(spectro_imagem_final_15, cmap='gray')


pylab.subplot(3, 7, 15)
pylab.axis('off')
pylab.title('Spectro Quadrantes')
pylab.imshow(spectro_imagem_original_quadrantes, cmap='gray')

pylab.subplot(3, 7, 16)
pylab.axis('off')
pylab.title('Spectros Ruidos')
pylab.imshow(spectro_ruido_quadrantes, cmap='gray')

pylab.subplot(3, 7, 17)
pylab.axis('off')
pylab.title('Spectros Ruidos 5')
pylab.imshow(spectro_5_quadrantes, cmap='gray')

pylab.subplot(3, 7, 18)
pylab.axis('off')
pylab.title('Spectros Ruidos 7')
pylab.imshow(spectro_7_quadrantes, cmap='gray')

pylab.subplot(3, 7, 19)
pylab.axis('off')
pylab.title('Spectros Ruidos 10')
pylab.imshow(spectro_10_quadrantes, cmap='gray')

pylab.subplot(3, 7, 20)
pylab.axis('off')
pylab.title('Spectros Ruidos 13')
pylab.imshow(spectro_13_quadrantes, cmap='gray')

pylab.subplot(3, 7, 21)
pylab.axis('off')
pylab.title('Spectros Ruidos 15')
pylab.imshow(spectro_15_quadrantes, cmap='gray')



pylab.show()



print('FIM TESTE FILTRO PASSA BAIXA')