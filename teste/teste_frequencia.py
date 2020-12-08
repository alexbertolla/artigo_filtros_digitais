import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import fftpack
from matplotlib import pylab
import cv2

def gerar_spectro(imagem):
    freq = np.fft.fft2(imagem)
    sfreq = np.fft.fftshift(freq)
    return (20 * np.log10(0.1 + sfreq)).real


def filtro_passa_alta(imagem, porcentagem_corte):
    freq = np.fft.fft2(imagem)
    sfreq = np.fft.fftshift(freq)
    spectro_original = (20 * np.log10(0.1 + sfreq)).real
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)
    lfreq = np.copy(sfreq)
    sfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0  # passa alta
    # lfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0 #passa baixa
    # sfreq -= lfreq
    imagem_passa_alta = np.fft.ifft2(np.fft.ifftshift(sfreq)).real
    return imagem_passa_alta

    #freq2 = np.fft.fft2(imagem_passa_alta)
    #ifreq2 = np.fft.ifftshift(freq2)
    #spectro_filtro = (20 * np.log10(0.1 + ifreq2)).real

    low_spectro = gerar_spectro(img_pb)
    #img_final = imagem_passa_alta + img_pb
    #print(img_pb)

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

caminho_imagem = '../imagens_ruido_gaussiano/hz_1235134-PPT.jpg'
caminho_imagem_ruidosa = '../imagens_ruido_gaussiano/hz_1235134-PPT.jpg'
imagem_original = img_as_float(imread(caminho_imagem, as_gray=True))
imagem_ruido = img_as_float(imread(caminho_imagem_ruidosa, as_gray=True))
linha, coluna = imagem_original.shape




#############SEPARA QUADRANTES#############
q1 = imagem_original[:int(linha / 2), :int(coluna / 2)]
q2 = imagem_original[:int(linha / 2), int(coluna / 2):]
q3 = imagem_original[int(linha / 2):, :int(coluna / 2)]
q4 = imagem_original[int(linha / 2):, int(coluna / 2):]

q1_ruido = imagem_ruido[:int(linha / 2), :int(coluna / 2)]
q2_ruido = imagem_ruido[:int(linha / 2), int(coluna / 2):]
q3_ruido = imagem_ruido[int(linha / 2):, :int(coluna / 2)]
q4_ruido = imagem_ruido[int(linha / 2):, int(coluna / 2):]

corte_pb = 0.1
q1_pb = aplicar_filtro_passa_baixa(q1_ruido, corte_pb)
q2_pb = aplicar_filtro_passa_baixa(q2_ruido, corte_pb)
q3_pb = aplicar_filtro_passa_baixa(q3_ruido, corte_pb)
q4_pb = aplicar_filtro_passa_baixa(q4_ruido, corte_pb)


corte_pa = 0.01
q1_pa = filtro_passa_alta(q1_pb, corte_pa)
q2_pa = filtro_passa_alta(q2_pb, corte_pa)
q3_pa = filtro_passa_alta(q3_pb, corte_pa)
q4_pa = filtro_passa_alta(q3_pb, corte_pa)

imagem_pb = np.zeros(imagem_original.shape)
imagem_pb[:int(linha / 2), :int(coluna / 2)] = q1_pb
imagem_pb[:int(linha / 2), int(coluna / 2):] = q2_pb
imagem_pb[int(linha / 2):, :int(coluna / 2)] = q3_pb
imagem_pb[int(linha / 2):, int(coluna / 2):] = q4_pb

imagem_pa = np.zeros(imagem_original.shape)
imagem_pa[:int(linha / 2), :int(coluna / 2)] = q1_pa
imagem_pa[:int(linha / 2), int(coluna / 2):] = q2_pa
imagem_pa[int(linha / 2):, :int(coluna / 2)] = q3_pa
imagem_pa[int(linha / 2):, int(coluna / 2):] = q4_pa
#############FIM SEPARA QUADRANTES#############

img_final = imagem_pb + imagem_pa
epectro = gerar_spectro(imagem_pa)
pylab.figure()
pylab.imshow(epectro, cmap='gray')
pylab.show()
exit()
#print(img_final*255)
#exit()
cv2.imshow('passa alta ' + str(corte_pa), imagem_pa)
cv2.imshow('passa baixa ' + str(corte_pb), imagem_pb)
cv2.imshow('imagem final ', epectro)
cv2.waitKey()
cv2.destroyAllWindows()

print('imagem_original: ' + str(imagem_original.dtype))

print('FIM TESTE FREQUENCIA')