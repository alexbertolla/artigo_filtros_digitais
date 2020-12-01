import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import fftpack


caminho_imagem_ruido = '../imagens_ruido_gaussiano/hz_1235134-PPT.jpg'
imagem_ruidosa = img_as_float(imread(caminho_imagem_ruido, as_gray=True))
linha, coluna = imagem_ruidosa.shape

print('Mediana U8 = ', str(np.median(img_as_ubyte(imagem_ruidosa))))

discrete_transform_imagem = fp.fft2(imagem_ruidosa)
mediana_frq = np.median(discrete_transform_imagem)
discrete_transform_01 = discrete_transform_imagem - (discrete_transform_imagem * 0.1)
print('Mediana FRQ = ', str(mediana_frq))
#print('Mediana 10% FRQ = ', str(np.median(discrete_transform_01)))

inversa = fp.ifft2(discrete_transform_imagem).real * 255
print('Mediana Inversa = ', str(np.median(inversa)))

inversa_01 = fp.ifft2(discrete_transform_01).real * 255
print('Mediana Inversa 10% = ', str(np.median(inversa_01)))

print('FIM TESTE FREQUENCIA')