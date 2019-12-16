import cv2 as cv
import matplotlib.pyplot as plt

from programa import add_ruido_gaussiano

imagem_original = cv.imread('imagem_original.jpg')
imagem_original = cv.cvtColor(imagem_original, cv.COLOR_BGR2RGB)
plt.figure()
plt.title('imagem_original')
plt.imshow(imagem_original)
plt.axis('off')

#imagem_ruido_sal_e_pimenta = add_ruido_sal_e_pimenta(imagem_original)
#imagem_ruido_sal_e_pimenta = cv.cvtColor(imagem_ruido_sal_e_pimenta, cv.COLOR_GRAY2RGB)
#plt.imsave('imagem_ruido_sal_e_pimenta.jpg', imagem_ruido_sal_e_pimenta)
#plt.figure()
#plt.title('imagem_ruido_sal_e_pimenta')
#plt.imshow(imagem_ruido_sal_e_pimenta)
#plt.axis('off')


imagem_ruido_gaussiano = add_ruido_gaussiano.add_ruido_gaussiano(imagem_original)
plt.imsave('imagem_ruido_gaussiano.jpg', imagem_ruido_gaussiano)
imagem_ruido_gaussiano = cv.imread('imagem_ruido_gaussiano.jpg')
imagem_ruido_gaussiano = cv.cvtColor(imagem_ruido_gaussiano, cv.COLOR_BGR2RGB)
#plt.imsave('imagem_ruido_gaussiano.jpg', imagem_ruido_gaussiano)
plt.figure()
plt.title('imagem_ruido_gaussiano')
plt.imshow(imagem_ruido_gaussiano)
plt.axis('off')


############################## FILTRO MEDIANA ###################################
#imagem_filtro_mediana_sp = cv.medianBlur(imagem_ruido_sal_e_pimenta, 3)
#plt.imsave('imagem_filtro_mediana_ruido_sal_e_pimenta.jpg', imagem_filtro_mediana_sp)
#plt.figure()
#plt.title('filtro mediana ruido sal e pimenta')
#plt.imshow(imagem_filtro_mediana_sp)
#plt.axis('off')

imagem_filtro_mediana_ruido_gaussiano = cv.medianBlur(imagem_ruido_gaussiano, 3)
plt.imsave('imagem_filtro_mediana_ruido_gaussiano.jpg', imagem_filtro_mediana_ruido_gaussiano)
plt.figure()
plt.title('filtro mediana ruido gaussiano')
plt.imshow(imagem_filtro_mediana_ruido_gaussiano)
plt.axis('off')
############################## FIM FILTRO MEDIANA ###################################

############################## FILTRO GAUSSIANO ###################################
#imagem_filtro_gaussiano_ruido_sal_e_pimenta = cv.GaussianBlur(imagem_ruido_sal_e_pimenta, (3, 3), 1)
#plt.imsave('imagem_filtro_gaussiano_ruido_sal_e_pimenta.jpg', imagem_filtro_gaussiano_ruido_sal_e_pimenta)
#plt.figure()
#plt.title('filtro gaussiano ruido sal e pimenta')
#plt.imshow(imagem_filtro_gaussiano_ruido_sal_e_pimenta)
#plt.axis('off')

imagem_filtro_gaussiano_ruido_gaussiano = cv.GaussianBlur(imagem_ruido_gaussiano, (3, 3), 1)
plt.imsave('imagem_filtro_gaussiano_ruido_gaussiano.jpg', imagem_filtro_gaussiano_ruido_gaussiano)
plt.figure()
plt.title('filtro gaussiano ruido gaussiano')
plt.imshow(imagem_filtro_gaussiano_ruido_gaussiano)
plt.axis('off')
############################## FIM FILTRO GAUSSIANO ###################################


#fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 4), sharex=True, sharey=True)
#plt.gray()
#axes[0][0].imshow(imagem_original), axes[0][0].axis('off'), axes[0][0].set_xlabel('original image')
#axes[0][1].imshow(imagem_ruido_gaussiano), axes[0][1].axis('off'), axes[0][1].set_xlabel('noisy')
#axes[0][2].imshow(imagem_filtro_mediana_g), axes[0][2].axis('off'), axes[0][2].set_xlabel('median filter')






plt.show()