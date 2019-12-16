import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage import data, img_as_float, color
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread, imsave, imshow


def mse(x, y):
    return np.linalg.norm(x - y)


imagem_original = cv.imread('imagem_original.jpg')
imagem_original = cv.cvtColor(imagem_original, cv.COLOR_BGR2GRAY)
imagem_original2 = cv.imread('imagem_original.jpg')


imagem_ruido_gaussiano = cv.imread('imagem_ruido_gaussiano.jpg')


imagem_filtro_gaussiano_ruido_gaussiano = cv.imread('imagem_filtro_gaussiano_ruido_gaussiano.jpg')
imagem_filtro_gaussiano_ruido_sal_e_pimenta = cv.imread('imagem_filtro_gaussiano_ruido_sal_e_pimenta.jpg')
imagem_filtro_gaussiano_ruido_gaussiano = cv.cvtColor(imagem_filtro_gaussiano_ruido_gaussiano, cv.COLOR_BGR2GRAY)
imagem_filtro_gaussiano_ruido_sal_e_pimenta = cv.cvtColor(imagem_filtro_gaussiano_ruido_sal_e_pimenta, cv.COLOR_BGR2GRAY)

imagem_filtro_mediana_ruido_gaussiano = cv.imread('imagem_filtro_mediana_ruido_gaussiano.jpg')
imagem_filtro_mediana_ruido_sal_e_pimenta = cv.imread('imagem_filtro_mediana_ruido_sal_e_pimenta.jpg')
imagem_filtro_mediana_ruido_gaussiano = cv.cvtColor(imagem_filtro_mediana_ruido_gaussiano, cv.COLOR_BGR2GRAY)
imagem_filtro_mediana_ruido_sal_e_pimenta = cv.cvtColor(imagem_filtro_mediana_ruido_sal_e_pimenta, cv.COLOR_BGR2GRAY)

imagem_filtro_wiener_ruido_gaussiano = cv.imread('imagem_filtro_wiener_ruido_gaussiano.jpg')
imagem_filtro_wiener_ruido_sal_e_pimenta = cv.imread('imagem_filtro_wiener_ruido_sal_e_pimenta.jpg')

imagem_filtro_wiener_ruido_gaussiano = cv.cvtColor(imagem_filtro_wiener_ruido_gaussiano, cv.COLOR_BGR2GRAY)
imagem_filtro_wiener_ruido_sal_e_pimenta = cv.cvtColor(imagem_filtro_wiener_ruido_sal_e_pimenta, cv.COLOR_BGR2GRAY)

#ssim_noise = ssim(img, img_noise, data_range=img_noise.max() - img_noise.min())
ssim_imagem_filtro_gaussiano_ruido_gaussiano = ssim(imagem_original, imagem_filtro_gaussiano_ruido_gaussiano)
ssim_imagem_filtro_gaussiano_ruido_sal_e_pimenta = ssim(imagem_original, imagem_filtro_gaussiano_ruido_sal_e_pimenta)

ssim_imagem_filtro_mediana_ruido_gaussiano = ssim(imagem_original, imagem_filtro_mediana_ruido_gaussiano)
ssim_imagem_filtro_mediana_ruido_sal_e_pimenta = ssim(imagem_original, imagem_filtro_mediana_ruido_sal_e_pimenta)

ssim_imagem_filtro_wiener_ruido_gaussiano = ssim(imagem_original, imagem_filtro_wiener_ruido_gaussiano)
ssim_imagem_filtro_wiener_ruido_sal_e_pimenta = ssim(imagem_original, imagem_filtro_wiener_ruido_sal_e_pimenta)

print('ssim_imagem_filtro_gaussiano_ruido_gaussiano: ', ssim_imagem_filtro_gaussiano_ruido_gaussiano)
print('ssim_imagem_filtro_mediana_ruido_gaussiano: ', ssim_imagem_filtro_mediana_ruido_gaussiano)
print('ssim_imagem_filtro_wiener_ruido_gaussiano: ', ssim_imagem_filtro_wiener_ruido_gaussiano)

print()

print('ssim_imagem_filtro_gaussiano_ruido_sal_e_pimenta: ', ssim_imagem_filtro_gaussiano_ruido_sal_e_pimenta)
print('ssim_imagem_filtro_mediana_ruido_sal_e_pimenta: ', ssim_imagem_filtro_mediana_ruido_sal_e_pimenta)
print('ssim_imagem_filtro_wiener_ruido_sal_e_pimenta: ', ssim_imagem_filtro_wiener_ruido_sal_e_pimenta)







#plt.figure()
#plt.gray()
#plt.imshow(imagem_original)


#plt.show()





print('FIM')