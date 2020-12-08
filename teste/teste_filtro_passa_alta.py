import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from matplotlib import pylab
import cv2 as cv

def filtro_passa_alta(imagem, porcentagem_corte):
    freq = fp.fft2(imagem)
    sfreq = fp.fftshift(freq)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)
    sfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0
    imagem_passa_alta = fp.ifft2(fp.ifftshift(sfreq)).real
    return imagem_passa_alta

caminho_imagem = '../banco_imagens/hz_1235134-PPT.jpg'
caminho_imagem_ruidosa = '../imagens_ruido_gaussiano/hz_1235134-PPT.jpg'

imagem_original = img_as_float(imread(caminho_imagem, as_gray=True))
imagem_ruido = img_as_float(imread(caminho_imagem_ruidosa, as_gray=True))

#############SEPARA QUADRANTES#############
linha, coluna = imagem_original.shape
q1 = imagem_original[:int(linha / 2), :int(coluna / 2)]
q2 = imagem_original[:int(linha / 2), int(coluna / 2):]
q3 = imagem_original[int(linha / 2):, :int(coluna / 2)]
q4 = imagem_original[int(linha / 2):, int(coluna / 2):]

q1_ruido = imagem_ruido[:int(linha / 2), :int(coluna / 2)]
q2_ruido = imagem_ruido[:int(linha / 2), int(coluna / 2):]
q3_ruido = imagem_ruido[int(linha / 2):, :int(coluna / 2)]
q4_ruido = imagem_ruido[int(linha / 2):, int(coluna / 2):]
#############FIM SEPARA QUADRANTES#############

lista_porcentagem_corte = ([0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
for corte in lista_porcentagem_corte:
    q1_filtro = filtro_passa_alta(q1_ruido, corte)
    q2_filtro = filtro_passa_alta(q2_ruido, corte)
    q3_filtro = filtro_passa_alta(q3_ruido, corte)
    q4_filtro = filtro_passa_alta(q4_ruido, corte)

    imagem_filtrada = np.zeros(imagem_original.shape)
    imagem_filtrada[:int(linha / 2), :int(coluna / 2)] = q1_filtro
    imagem_filtrada[:int(linha / 2), int(coluna / 2):] = q2_filtro
    imagem_filtrada[int(linha / 2):, :int(coluna / 2)] = q3_filtro
    imagem_filtrada[int(linha / 2):, int(coluna / 2):] = q4_filtro

    imagem_filtrada = img_as_ubyte(imagem_filtrada)

    cv.imshow('imagem ' + str(corte), imagem_filtrada)

cv.waitKey()
cv.destroyAllWindows()



print('FIM TESTE PASSA ALTA')