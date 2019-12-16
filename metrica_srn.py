import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float, color
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread, imsave, imshow
import cv2 as cv



def mse(vref, vcmp):
    """
    Compute Mean Squared Error (MSE) between two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      MSE between `vref` and `vcmp`
    """

    r = np.asarray(vref, dtype=np.float32).ravel()
    c = np.asarray(vcmp, dtype=np.float32).ravel()
    return np.mean(np.abs(r - c)**2)




def snr(vref, vcmp):
    """
    Compute Signal to Noise Ratio (SNR) of two images.

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image

    Returns
    -------
    x : float
      SNR of `vcmp` with respect to `vref`
    """

    dv = np.var(vref)
    with np.errstate(divide='ignore'):
        rt = dv / mse(vref, vcmp)
    return 10.0 * np.log10(rt)


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

snr_imagem_filtro_gaussiano_ruido_gaussiano = snr(imagem_original, imagem_filtro_gaussiano_ruido_gaussiano)
snr_imagem_filtro_gaussiano_ruido_sal_e_pimenta = snr(imagem_original, imagem_filtro_gaussiano_ruido_sal_e_pimenta)

snr_imagem_filtro_mediana_ruido_gaussiano = snr(imagem_original, imagem_filtro_mediana_ruido_gaussiano)
snr_imagem_filtro_mediana_ruido_sal_e_pimenta = snr(imagem_original, imagem_filtro_mediana_ruido_sal_e_pimenta)

snr_imagem_filtro_wiener_ruido_gaussiano = snr(imagem_original, imagem_filtro_wiener_ruido_gaussiano)
snr_imagem_filtro_wiener_ruido_sal_e_pimenta = snr(imagem_original, imagem_filtro_wiener_ruido_sal_e_pimenta)

print('snr_imagem_filtro_gaussiano_ruido_gaussiano = ', snr_imagem_filtro_gaussiano_ruido_gaussiano)
print('snr_imagem_filtro_mediana_ruido_gaussiano = ', snr_imagem_filtro_mediana_ruido_gaussiano)
print('snr_imagem_filtro_wiener_ruido_gaussiano = ', snr_imagem_filtro_wiener_ruido_gaussiano)

print('snr_imagem_filtro_gaussiano_ruido_sal_e_pimenta = ', snr_imagem_filtro_gaussiano_ruido_sal_e_pimenta)
print('snr_imagem_filtro_mediana_ruido_sal_e_pimenta = ', snr_imagem_filtro_mediana_ruido_sal_e_pimenta)
print('snr_imagem_filtro_wiener_ruido_sal_e_pimenta = ', snr_imagem_filtro_wiener_ruido_sal_e_pimenta)

print(snr_imagem_filtro_gaussiano_ruido_gaussiano - snr_imagem_filtro_gaussiano_ruido_sal_e_pimenta)
print(snr_imagem_filtro_mediana_ruido_gaussiano - snr_imagem_filtro_mediana_ruido_sal_e_pimenta)
print(snr_imagem_filtro_wiener_ruido_gaussiano - snr_imagem_filtro_wiener_ruido_sal_e_pimenta)

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 4), sharex=True, sharey=True)
plt.gray()
plt.axis('off')
axes[0][0].imshow(imagem_original), axes[0][0].axis('off'), axes[0][0].set_xlabel('original image')
axes[0][1].imshow(imagem_filtro_gaussiano_ruido_gaussiano), axes[0][1].axis('on'), axes[0][1].set_xlabel('imagem_filtro_gaussiano_ruido_gaussiano')
axes[0][2].imshow(imagem_filtro_mediana_ruido_gaussiano), axes[0][2].axis('on'), axes[0][2].set_xlabel('imagem_filtro_mediana_ruido_gaussiano')
axes[0][3].imshow(imagem_filtro_wiener_ruido_gaussiano), axes[0][3].axis('on'), axes[0][3].set_xlabel('imagem_filtro_wiener_ruido_gaussiano')

axes[1][0].imshow(imagem_original), axes[1][0].axis('off'), axes[1][0].set_xlabel('original image')
axes[1][1].imshow(imagem_filtro_gaussiano_ruido_sal_e_pimenta), axes[1][1].axis('on'), axes[1][1].set_xlabel('imagem_filtro_gaussiano_ruido_sal_e_pimenta')
axes[1][2].imshow(imagem_filtro_mediana_ruido_sal_e_pimenta), axes[1][2].axis('on'), axes[1][2].set_xlabel('imagem_filtro_mediana_ruido_sal_e_pimenta')
axes[1][3].imshow(imagem_filtro_wiener_ruido_sal_e_pimenta), axes[1][3].axis('on'), axes[1][3].set_xlabel('imagem_filtro_wiener_ruido_sal_e_pimenta')
plt.show()

cv.imshow('1', imagem_filtro_wiener_ruido_gaussiano)
cv.imshow('2', imagem_filtro_wiener_ruido_sal_e_pimenta)

cv.waitKey()