import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from matplotlib import pylab
import cv2 as cv

def gerar_espectro(imagem):
    freq = fp.fft2(imagem)
    sfreq = fp.fftshift(freq)
    return (20 * np.log10(0.1 + sfreq)).real


def filtro_passa_alta(imagem, porcentagem_corte):
    freq = fp.fft2(imagem)
    sfreq = fp.fftshift(freq)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)
    fcorte = int(half_w * porcentagem_corte)
    sfreq[half_w - fcorte:half_w + fcorte + 1, half_h - fcorte:half_h + fcorte + 1] = 0

    imagem_passa_alta = np.clip(fp.ifft2(fp.ifftshift(sfreq)).real, 0, 1)
    espectro_filtro = (20 * np.log10(0.1 + sfreq)).real

    return imagem_passa_alta, espectro_filtro

caminho_imagem = '../banco_imagens/sf_5507028-PPT.jpg'
caminho_imagem_ruidosa = '../imagens_ruido_gaussiano/sf_5507028-PPT.jpg'

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

espectro_original = gerar_espectro(imagem_original)
q1_espectro = gerar_espectro(q1)
q2_espectro = gerar_espectro(q2)
q3_espectro = gerar_espectro(q3)
q4_espectro = gerar_espectro(q4)
espectros_originais = np.zeros((imagem_original.shape))
espectros_originais[:int(linha / 2), :int(coluna / 2)] = q1_espectro
espectros_originais[:int(linha / 2), int(coluna / 2):] = q2_espectro
espectros_originais[int(linha / 2):, :int(coluna / 2)] = q3_espectro
espectros_originais[int(linha / 2):, int(coluna / 2):] = q4_espectro

espectro_ruido = gerar_espectro(imagem_ruido)
q1_espectro_ruido = gerar_espectro(q1_ruido)
q2_espectro_ruido = gerar_espectro(q2_ruido)
q3_espectro_ruido = gerar_espectro(q3_ruido)
q4_espectro_ruido = gerar_espectro(q4_ruido)
espectros_ruidos = np.zeros((imagem_ruido.shape))
espectros_ruidos[:int(linha / 2), :int(coluna / 2)] = q1_espectro_ruido
espectros_ruidos[:int(linha / 2), int(coluna / 2):] = q2_espectro_ruido
espectros_ruidos[int(linha / 2):, :int(coluna / 2)] = q3_espectro_ruido
espectros_ruidos[int(linha / 2):, int(coluna / 2):] = q4_espectro_ruido

pltL = 3
pltC = 8
pltPI = 1
pltPSI = 9
pltPF = 17

pylab.figure()
pylab.subplot(pltL, pltC, pltPI)
pylab.axis('off')
pylab.title('Imagem Original')
pylab.imshow(imagem_original, cmap='gray')
pltPI += 1

pylab.subplot(pltL, pltC, pltPI)
pylab.axis('off')
pylab.title('Imagem Ruído')
pylab.imshow(imagem_ruido, cmap='gray')
pltPI += 1

pylab.subplot(pltL, pltC, pltPSI)
pylab.axis('off')
#pylab.title('Original')
pylab.imshow(espectro_original, cmap='gray')
pltPSI += 1

pylab.subplot(pltL, pltC, pltPF)
pylab.axis('off')
#pylab.title('Espectros Quadrantes')
pylab.imshow(espectros_originais, cmap='gray')
pltPF += 1

pylab.subplot(pltL, pltC, pltPSI)
pylab.axis('off')
#pylab.title('Ruído')
pylab.imshow(espectro_ruido, cmap='gray')
pltPSI += 1

pylab.subplot(pltL, pltC, pltPF)
pylab.axis('off')
#pylab.title('Espectros Quadrantes Ruídos')
pylab.imshow(espectros_ruidos, cmap='gray')
pltPF += 1

lista_porcentagem_corte = ([0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
for corte in lista_porcentagem_corte:
    q1_filtro, espectro_filtro_q1 = filtro_passa_alta(q1_ruido, corte)
    q2_filtro, espectro_filtro_q2 = filtro_passa_alta(q2_ruido, corte)
    q3_filtro, espectro_filtro_q3 = filtro_passa_alta(q3_ruido, corte)
    q4_filtro, espectro_filtro_q4 = filtro_passa_alta(q4_ruido, corte)

    imagem_fitro_pa, espectro_pa = filtro_passa_alta(imagem_ruido, corte)

    imagem_filtrada = np.zeros(imagem_original.shape)
    imagem_filtrada[:int(linha / 2), :int(coluna / 2)] = q1_filtro
    imagem_filtrada[:int(linha / 2), int(coluna / 2):] = q2_filtro
    imagem_filtrada[int(linha / 2):, :int(coluna / 2)] = q3_filtro
    imagem_filtrada[int(linha / 2):, int(coluna / 2):] = q4_filtro

    espectro_filtro = np.zeros(imagem_filtrada.shape)
    espectro_filtro[:int(linha / 2), :int(coluna / 2)] = espectro_filtro_q1
    espectro_filtro[:int(linha / 2), int(coluna / 2):] = espectro_filtro_q2
    espectro_filtro[int(linha / 2):, :int(coluna / 2)] = espectro_filtro_q3
    espectro_filtro[int(linha / 2):, int(coluna / 2):] = espectro_filtro_q4

    print(np.min(imagem_filtrada))
    print()
    imagem_filtrada = img_as_ubyte(abs(imagem_filtrada))
    print(imagem_filtrada)

    pylab.subplot(pltL, pltC, pltPI)
    pylab.axis('off')
    pylab.title('F.P.A ' + str(corte*100) + '%')
    pylab.imshow(img_as_ubyte(imagem_filtrada), cmap='gray')
    pltPI += 1

    pylab.subplot(pltL, pltC, pltPSI)
    pylab.axis('off')
    # pylab.title('Ruído')
    pylab.imshow(espectro_pa, cmap='gray')
    pltPSI += 1

    pylab.subplot(pltL, pltC, pltPF)
    pylab.axis('off')
    # pylab.title('Espectros Quadrantes Ruídos')
    pylab.imshow(espectro_filtro, cmap='gray')
    pltPF += 1
    #exit()

    #cv.imshow('imagem ' + str(corte), imagem_filtrada)

pylab.show()
cv.waitKey()
cv.destroyAllWindows()



print('FIM TESTE PASSA ALTA')