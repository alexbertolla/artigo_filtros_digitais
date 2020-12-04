import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import fftpack
from matplotlib import pylab
from skimage.filters import gaussian
import os
import shutil

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, mode='gaussian', var=sigma)
    return imagem_ruido_gaussiano

def gerar_spectro(imagem):
    discrete_transform_imagem = fp.fft2(imagem)
    (w, h) = discrete_transform_imagem.shape
    half_w, half_h = int(w / 2), int(h / 2)
    shift_frq = fftpack.fftshift(discrete_transform_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

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


lista_porcentagem_corte = ([0.05, 0.10, 0.15])

caminho_imagem_original = '../banco_imagens/hz_1235134-PPT.jpg'
imagem_original = img_as_float(imread(caminho_imagem_original, as_gray=True))
linha, coluna = imagem_original.shape

quadrante_1 = imagem_original[:int(linha / 2), :int(coluna / 2)]
quadrante_2 = imagem_original[:int(linha / 2), int(coluna / 2):]
quadrante_3 = imagem_original[int(linha / 2):, :int(coluna / 2)]
quadrante_4 = imagem_original[int(linha / 2):, int(coluna / 2):]


quadrante_ruido_1 = add_ruido_gaussiano(quadrante_1, 0.00)
quadrante_ruido_2 = add_ruido_gaussiano(quadrante_2, 0.05)
quadrante_ruido_3 = add_ruido_gaussiano(quadrante_3, 0.10)
quadrante_ruido_4 = add_ruido_gaussiano(quadrante_4, 0.15)

imagem_ruido_gaussiano = np.zeros(imagem_original.shape)
imagem_ruido_gaussiano[:int(linha / 2), :int(coluna / 2)] = quadrante_ruido_1
imagem_ruido_gaussiano[:int(linha / 2), int(coluna / 2):] = quadrante_ruido_2
imagem_ruido_gaussiano[int(linha / 2):, :int(coluna / 2)] = quadrante_ruido_3
imagem_ruido_gaussiano[int(linha / 2):, int(coluna / 2):] = quadrante_ruido_4

imagem_final_5 = np.zeros(imagem_original.shape)
imagem_final_10 = np.zeros(imagem_original.shape)
imagem_final_15 = np.zeros(imagem_original.shape)

imagem_final_5[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_alta(quadrante_ruido_1, 0.05)
imagem_final_5[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_alta(quadrante_ruido_2, 0.05)
imagem_final_5[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_alta(quadrante_ruido_3, 0.05)
imagem_final_5[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_alta(quadrante_ruido_4, 0.05)

imagem_final_10[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_alta(quadrante_ruido_1, 0.10)
imagem_final_10[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_alta(quadrante_ruido_2, 0.10)
imagem_final_10[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_alta(quadrante_ruido_3, 0.10)
imagem_final_10[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_alta(quadrante_ruido_4, 0.10)

imagem_final_15[:int(linha / 2), :int(coluna / 2)] = aplicar_filtro_passa_alta(quadrante_ruido_1, 0.15)
imagem_final_15[:int(linha / 2), int(coluna / 2):] = aplicar_filtro_passa_alta(quadrante_ruido_2, 0.15)
imagem_final_15[int(linha / 2):, :int(coluna / 2)] = aplicar_filtro_passa_alta(quadrante_ruido_3, 0.15)
imagem_final_15[int(linha / 2):, int(coluna / 2):] = aplicar_filtro_passa_alta(quadrante_ruido_4, 0.15)

imagem_final_5 = img_as_ubyte(imagem_final_5)
imagem_final_10 = img_as_ubyte(imagem_final_10)
imagem_final_15 = img_as_ubyte(imagem_final_15)


spectro_imagem_original = gerar_spectro(imagem_original)
spectro_imagem_ruidosa = gerar_spectro(imagem_ruido_gaussiano)
spectro_imagem_final_5 = gerar_spectro(imagem_final_5)
spectro_imagem_final_10 = gerar_spectro(imagem_final_10)
spectro_imagem_final_15 = gerar_spectro(imagem_final_15)

pylab.figure()
pylab.subplot(2, 5, 1)
pylab.axis('off')
pylab.title('Imagem Original')
pylab.imshow(imagem_original, cmap='gray')

pylab.subplot(2, 5, 2)
pylab.axis('off')
pylab.title('Imagem Ruídosa')
pylab.imshow(imagem_ruido_gaussiano, cmap='gray')

pylab.subplot(2, 5, 3)
pylab.axis('off')
pylab.title('Imagem Filtrada 5%')
pylab.imshow(imagem_final_5, cmap='gray')

pylab.subplot(2, 5, 4)
pylab.axis('off')
pylab.title('Imagem Filtrada 10%')
pylab.imshow(imagem_final_10, cmap='gray')

pylab.subplot(2, 5, 5)
pylab.axis('off')
pylab.title('Imagem Filtrada 15%')
pylab.imshow(imagem_final_15, cmap='gray')

pylab.subplot(2, 5, 6)
pylab.axis('off')
pylab.title('Spectro Imagem Original')
pylab.imshow(spectro_imagem_original, cmap='gray')

pylab.subplot(2, 5, 7)
pylab.axis('off')
pylab.title('Spectro Imagem Ruidosa')
pylab.imshow(spectro_imagem_ruidosa, cmap='gray')

pylab.subplot(2, 5, 8)
pylab.axis('off')
pylab.title('Spectro 5%')
pylab.imshow(spectro_imagem_final_5, cmap='gray')

pylab.subplot(2, 5, 9)
pylab.axis('off')
pylab.title('Spectro 10%')
pylab.imshow(spectro_imagem_final_10, cmap='gray')

pylab.subplot(2, 5, 10)
pylab.axis('off')
pylab.title('Spectro 15%')
pylab.imshow(spectro_imagem_final_15, cmap='gray')

pylab.show()


print('FIM TESTE FILTRO PASSA ALATA')