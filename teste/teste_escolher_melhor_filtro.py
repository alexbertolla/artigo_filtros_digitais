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

def add_ruido_gaussiano(imagem_original, sigma):
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano

def gerar_imagem_ruidosa(q1, q2, q3, q4, modelo_imagem):
    imagem_ruidosa = np.zeros(modelo_imagem.shape)
    imagem_ruidosa[:int(l / 2), :int(c / 2)] = q1_ruidoso
    imagem_ruidosa[:int(l / 2), int(c / 2):] = q2_ruidoso
    imagem_ruidosa[int(l / 2):, :int(c / 2)] = q3_ruidoso
    imagem_ruidosa[int(l / 2):, int(c / 2):] = q4_ruidoso
    return imagem_ruidosa



caminho_imagem = '../banco_imagens/hz_1235134-PPT.jpg'
imagem_original = img_as_float(imread(caminho_imagem, as_gray=True))
lista_instensidade_ruido = [0.00, 0.05, 0.10, 0.15]

l, c = imagem_original.shape

q1_original = imagem_original[:int(l/2), :int(c/2)]
q2_original = imagem_original[:int(l/2), int(c/2):]
q3_original = imagem_original[int(l/2):, :int(c/2)]
q4_original = imagem_original[int(l/2):, int(c/2):]

pylab.subplot(1, 4, 1), pylab.title('Imagem Original')
pylab.axis('off')
pylab.imshow(imagem_original, cmap='gray')


ruido_q1 = random.choice(lista_instensidade_ruido)
ruido_q2 = random.choice(lista_instensidade_ruido)
ruido_q3 = random.choice(lista_instensidade_ruido)
ruido_q4 = random.choice(lista_instensidade_ruido)
print('Intensidades Ruído Imagem 1')
print('Q1: ' + str(ruido_q1))
print('Q2: ' + str(ruido_q2))
print('Q3: ' + str(ruido_q3))
print('Q4: ' + str(ruido_q4))
print()

q1_ruidoso = add_ruido_gaussiano(q1_original, ruido_q1)
q2_ruidoso = add_ruido_gaussiano(q2_original, ruido_q2)
q3_ruidoso = add_ruido_gaussiano(q3_original, ruido_q3)
q4_ruidoso = add_ruido_gaussiano(q4_original, ruido_q4)

imagem_ruidosa_1 = gerar_imagem_ruidosa(q1_original, q2_original, q3_original, q4_original, imagem_original)
pylab.subplot(1, 4, 2), pylab.title('Q1:'+str(ruido_q1)+' | Q2:'+str(ruido_q2)+' | Q3:'+str(ruido_q3)+' | Q:4'+str(ruido_q4))
pylab.axis('off')
pylab.imshow(imagem_ruidosa_1, cmap='gray')

ruido_q1 = random.choice(lista_instensidade_ruido)
ruido_q2 = random.choice(lista_instensidade_ruido)
ruido_q3 = random.choice(lista_instensidade_ruido)
ruido_q4 = random.choice(lista_instensidade_ruido)
print('Intensidades Ruído Imagem 2')
print('Q1: ' + str(ruido_q1))
print('Q2: ' + str(ruido_q2))
print('Q3: ' + str(ruido_q3))
print('Q4: ' + str(ruido_q4))
print()

q1_ruidoso = add_ruido_gaussiano(q1_original, ruido_q1)
q2_ruidoso = add_ruido_gaussiano(q2_original, ruido_q2)
q3_ruidoso = add_ruido_gaussiano(q3_original, ruido_q3)
q4_ruidoso = add_ruido_gaussiano(q4_original, ruido_q4)

imagem_ruidosa_2 = gerar_imagem_ruidosa(q1_original, q2_original, q3_original, q4_original, imagem_original)
pylab.subplot(1, 4, 3), pylab.title('Q1:'+str(ruido_q1)+' | Q2:'+str(ruido_q2)+' | Q3:'+str(ruido_q3)+' | Q:4'+str(ruido_q4))
pylab.axis('off')
pylab.imshow(imagem_ruidosa_2, cmap='gray')

ruido_q1 = random.choice(lista_instensidade_ruido)
ruido_q2 = random.choice(lista_instensidade_ruido)
ruido_q3 = random.choice(lista_instensidade_ruido)
ruido_q4 = random.choice(lista_instensidade_ruido)
print('Intensidades Ruído Imagem 3')
print('Q1: ' + str(ruido_q1))
print('Q2: ' + str(ruido_q2))
print('Q3: ' + str(ruido_q3))
print('Q4: ' + str(ruido_q4))
print()

q1_ruidoso = add_ruido_gaussiano(q1_original, ruido_q1)
q2_ruidoso = add_ruido_gaussiano(q2_original, ruido_q2)
q3_ruidoso = add_ruido_gaussiano(q3_original, ruido_q3)
q4_ruidoso = add_ruido_gaussiano(q4_original, ruido_q4)

imagem_ruidosa_3 = gerar_imagem_ruidosa(q1_original, q2_original, q3_original, q4_original, imagem_original)
pylab.subplot(1, 4, 4), pylab.title('Q1:'+str(ruido_q1)+' | Q2:'+str(ruido_q2)+' | Q3:'+str(ruido_q3)+' | Q:4'+str(ruido_q4))
pylab.axis('off')
pylab.imshow(imagem_ruidosa_3, cmap='gray')

pylab.show()

print('FIM ESCOLHER MELHOR FILTRO')