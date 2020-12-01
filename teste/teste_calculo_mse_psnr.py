import numpy as np
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import os
import shutil

def calcular_mse(img_ruidosa, img_filtrada):
    return round(mse(img_ruidosa, img_filtrada), 2)

def calcular_psnr(img_ruidosa, img_filtrada):
    return round(psnr(img_ruidosa, img_filtrada), 2)


caminho_imagem_ruidosa = '../imagens_ruido_gaussiano/hz_1235134-PPT.jpg'
caminho_imagem_filtrada = '../imagens_filtro_passa_baixa/0.1/hz_1235134-PPT.jpg'

imagem_ruidosa = imread(caminho_imagem_ruidosa, as_gray=True)
imagem_filtrada = imread(caminho_imagem_filtrada, as_gray=True)

linha, coluna = imagem_ruidosa.shape
q_1_ruidoso = imagem_ruidosa[:int(linha / 2), :int(coluna / 2)]
q_2_ruidoso = imagem_ruidosa[:int(linha / 2), int(coluna / 2):]
q_3_ruidoso = imagem_ruidosa[int(linha / 2):, :int(coluna / 2)]
q_4_ruidoso = imagem_ruidosa[int(linha / 2):, int(coluna / 2):]

q_1_filtrado = imagem_filtrada[:int(linha / 2), :int(coluna / 2)]
q_2_filtrado = imagem_filtrada[:int(linha / 2), int(coluna / 2):]
q_3_filtrado = imagem_filtrada[int(linha / 2):, :int(coluna / 2)]
q_4_filtrado = imagem_filtrada[int(linha / 2):, int(coluna / 2):]

array_mse = []
array_mse.append(calcular_mse(q_1_ruidoso, q_1_filtrado))
array_mse.append(calcular_mse(q_2_ruidoso, q_2_filtrado))
array_mse.append(calcular_mse(q_3_ruidoso, q_3_filtrado))
array_mse.append(calcular_mse(q_4_ruidoso, q_4_filtrado))
print('Valores MSE: ', str(array_mse))
print('Mediana MSE: ', str(np.median(array_mse)))

array_psnr = []
array_psnr.append(calcular_psnr(q_1_ruidoso, q_1_filtrado))
array_psnr.append(calcular_psnr(q_2_ruidoso, q_2_filtrado))
array_psnr.append(calcular_psnr(q_3_ruidoso, q_3_filtrado))
array_psnr.append(calcular_psnr(q_4_ruidoso, q_4_filtrado))
print('Valores PSNR: ', str(array_psnr))
print('Mediana PSNR: ', str(np.median(array_psnr)))

print('FIM TESTE MSR E PSNR')