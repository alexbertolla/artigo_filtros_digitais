from skimage import img_as_ubyte, img_as_float
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from sporco.metric import snr
from matplotlib import pylab

def calcular_snr(imagem_original, imagem_ruidosa):
    return snr(imagem_original, imagem_ruidosa)

def calcular_mse(imagem_original, imagem_ruidosa):
    return mean_squared_error(imagem_original, imagem_ruidosa)

lena = 'imagem_lena.png'

imagem_original = imread(lena, as_gray=True)
imagem_ruido_gaussiano = imread('imagem_ruido_gaussiano.jpg', as_gray=True)
imagem_ruido_spekle = imread('imagem_ruido_spekle.jpg', as_gray=True)
imagem_ruido_sal_e_pimenta = imread('imagem_ruido_sal_e_pimenta.jpg', as_gray=True)

snr_ruido_gaussiano = calcular_snr(imagem_original, imagem_ruido_gaussiano)
snr_ruido_spekle = calcular_snr(imagem_original, imagem_ruido_spekle)
snr_ruido_sal_e_pimenta = calcular_snr(imagem_original, imagem_ruido_sal_e_pimenta)

print('SNR RUÍDO GAUSSIANO = ', snr_ruido_gaussiano)
print('SNR RUÍDO SPEKLE = ', snr_ruido_spekle)
print('SNR RUÍDO SAL E PIMENTA = ', snr_ruido_sal_e_pimenta)



mse_ruido_gaussiano = calcular_mse(imagem_original, imagem_ruido_gaussiano)
mse_ruido_spekle = calcular_mse(imagem_original, imagem_ruido_spekle)
mse_ruido_sal_e_pimenta = calcular_mse(imagem_original, imagem_ruido_sal_e_pimenta)

print('SNR RUÍDO GAUSSIANO = ', snr_ruido_gaussiano)
print('SNR RUÍDO SPEKLE = ', snr_ruido_spekle)
print('SNR RUÍDO SAL E PIMENTA = ', snr_ruido_sal_e_pimenta)

print('MSE RUÍDO GAUSSIANO = ', mse_ruido_gaussiano)
print('MSE RUÍDO SPEKLE = ', mse_ruido_spekle)
print('MSE RUÍDO SAL E PIMENTA = ', mse_ruido_sal_e_pimenta)



pylab.figure()
pylab.subplot(1, 4, 1), pylab.imshow(imagem_original, cmap='gray')
pylab.subplot(1, 4, 2), pylab.imshow(imagem_ruido_gaussiano, cmap='gray')
pylab.subplot(1, 4, 3), pylab.imshow(imagem_ruido_spekle, cmap='gray')
pylab.subplot(1, 4, 4), pylab.imshow(imagem_ruido_sal_e_pimenta, cmap='gray')

pylab.show()

print('FIM METRICAS')