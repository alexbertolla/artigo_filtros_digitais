import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

from skimage import color, restoration

#from add_ruido_branco import add_ruido_branco

imagem_original = cv.imread('imagem_original.jpg')
imagem_original = cv.cvtColor(imagem_original, cv.COLOR_BGR2RGB)
plt.figure()
plt.title('imagem_original')
plt.imshow(imagem_original)
plt.axis('off')

n = 3
psf = np.ones((n, n)) / n**2

#imagem_ruidosa_livro = conv2(imagem_original, psf, 'same')
#imagem_ruidosa_livro += 5 * np.std(imagem_original.shape) * np.random.standard_normal(imagem_original.shape)

imagem_ruido_gaussiano = cv.imread('imagem_ruido_gaussiano.jpg')
imagem_ruido_gaussiano = cv.cvtColor(imagem_ruido_gaussiano, cv.COLOR_BGR2RGB)
plt.figure()
plt.title('imagem_ruido_gaussiano')
plt.imshow(imagem_ruido_gaussiano)
plt.axis('off')
#imagem_ruido_gaussiano = add_ruido_gaussiano(imagem_original)
#imagem_ruido_sal_e_pimenta = color.rgb2gray(imread('imagem_ruido_sal_e_pimenta.jpg'))



#imagem_livro_restaurada, _ = restoration.unsupervised_wiener(imagem_ruidosa_livro, psf)

imagem_filtro_wiener_ruido_gaussiano, _ = restoration.unsupervised_wiener(color.rgb2gray(imagem_ruido_gaussiano), psf)
plt.imsave('imagem_filtro_wiener_ruido_gaussiano.jpg', imagem_filtro_wiener_ruido_gaussiano)
#imagem_filtro_wiener_ruido_gaussiano = color.rgb2gray(imagem_filtro_wiener_ruido_gaussiano)
plt.figure()
plt.gray()
plt.title('imagem_filtro_wiener_ruido_gaussiano')
plt.imshow(imagem_filtro_wiener_ruido_gaussiano)
plt.axis('off')


imagem_filtro_wiener_ruido_gaussiano = cv.imread('imagem_filtro_wiener_ruido_gaussiano.jpg', 0)
#imagem_filtro_wiener_ruido_gaussiano = cv.cvtColor(imagem_filtro_wiener_ruido_gaussiano, cv.COLOR_BGR2RGB)
plt.imsave('imagem_filtro_wiener_ruido_gaussiano.jpg', imagem_filtro_wiener_ruido_gaussiano)

imagem_ruido_sal_e_pimenta = cv.imread('imagem_ruido_sal_e_pimenta.jpg')
imagem_filtro_wiener_ruido_sal_e_pimenta, _ = restoration.unsupervised_wiener(color.rgb2gray(imagem_ruido_sal_e_pimenta), psf)
plt.imsave('imagem_filtro_wiener_ruido_sal_e_pimenta.jpg', imagem_filtro_wiener_ruido_sal_e_pimenta)
plt.figure()
plt.gray()
plt.title('imagem_filtro_wiener_ruido_sal_e_pimenta')
plt.imshow(imagem_filtro_wiener_ruido_sal_e_pimenta)
plt.axis('off')
#imagem_sal_e_pimenta_restaurada, _ = restoration.unsupervised_wiener(imagem_ruido_sal_e_pimenta, psf)
#plt.imsave('imagem_filtro_wiener_ruido_sal_e_pimenta.jpg', imagem_sal_e_pimenta_restaurada)
#imagem_filtro_wiener_ruido_sal_e_pimenta = cv.imread('imagem_filtro_wiener_ruido_sal_e_pimenta.jpg', 0)
#print(imagem_filtro_wiener_ruido_sal_e_pimenta.shape)
#imagem_filtro_wiener_ruido_sal_e_pimenta = cv.cvtColor(imagem_filtro_wiener_ruido_sal_e_pimenta, cv.COLOR_RGB2GRAY)
#print(imagem_filtro_wiener_ruido_sal_e_pimenta.shape)
#plt.gray()
#plt.imsave('imagem_filtro_wiener_ruido_sal_e_pimenta.jpg', imagem_filtro_wiener_ruido_sal_e_pimenta)

#fig, axes = pylab.subplots(nrows=2, ncols=3, figsize=(15, 4), sharex=True, sharey=True)
#pylab.gray()
#pylab.axis('off')
#axes[0][0].imshow(imagem_original), axes[0][0].axis('off'), axes[0][0].set_xlabel('original image')
#axes[0][1].imshow(imagem_ruido_sal_e_pimenta), axes[0][1].axis('off'), axes[0][1].set_xlabel('Gaussian  noisy')
#axes[0][2].imshow(imagem_sal_e_pimenta_restaurada), axes[0][2].axis('off'), axes[0][2].set_xlabel('SP  noisy')


plt.show()

print('FIM')