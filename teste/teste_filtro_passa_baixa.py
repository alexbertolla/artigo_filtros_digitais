import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave

from scipy import fftpack
import cv2 as cv


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

caminho_imagem_ruidosa = '../imagens_ruido_gaussiano/hz_1235134-PPT.jpg'
imagem_ruido = img_as_float(imread(caminho_imagem_ruidosa, as_gray=True))

#############SEPARA QUADRANTES#############
linha, coluna = imagem_ruido.shape

q1_ruido = imagem_ruido[:int(linha / 2), :int(coluna / 2)]
q2_ruido = imagem_ruido[:int(linha / 2), int(coluna / 2):]
q3_ruido = imagem_ruido[int(linha / 2):, :int(coluna / 2)]
q4_ruido = imagem_ruido[int(linha / 2):, int(coluna / 2):]

#############FIM SEPARA QUADRANTES#############

lista_porcentagem_corte = ([0.05, 0.07, 0.10, 0.13, 0.15])
for porcentagem_corte in lista_porcentagem_corte:
    print(porcentagem_corte)
    q1_filtro = filtro_passa_baixa(q1_ruido, porcentagem_corte)
    q2_filtro = filtro_passa_baixa(q2_ruido, porcentagem_corte)
    q3_filtro = filtro_passa_baixa(q3_ruido, porcentagem_corte)
    q4_filtro = filtro_passa_baixa(q4_ruido, porcentagem_corte)

    imagem_filtrada = np.zeros(imagem_ruido.shape)
    imagem_filtrada[:int(linha / 2), :int(coluna / 2)] = q1_filtro
    imagem_filtrada[:int(linha / 2), int(coluna / 2):] = q2_filtro
    imagem_filtrada[int(linha / 2):, :int(coluna / 2)] = q3_filtro
    imagem_filtrada[int(linha / 2):, int(coluna / 2):] = q4_filtro

    imagem_filtrada = img_as_ubyte(imagem_filtrada)

    cv.imshow('imagem ' + str(porcentagem_corte), imagem_filtrada)

cv.waitKey()
cv.destroyAllWindows()

print('FIM TESTE FILTRO PASSA BAIXA')