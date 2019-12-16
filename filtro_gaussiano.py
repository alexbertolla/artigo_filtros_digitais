import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from add_ruido_gaussiano import add_ruido_gaussiano
from add_ruido_sal_e_pimenta import add_ruido_sal_e_pimenta

nome_imagem_original = "imagem_original.jpg"
imagem_original = cv.imread(nome_imagem_original, 0)
lin, col = imagem_original.shape

arr = []


imagem_original = cv.cvtColor(imagem_original, cv.COLOR_BGR2RGB)
#imagem_original = cv.cvtColor(imagem_original, cv.COLOR_RGB2GRAY)
plt.figure()
plt.axis("off")
plt.imshow(imagem_original, cmap='gray')


ruidos_gaussiano = add_ruido_gaussiano(imagem_original)

plt.figure()
plt.axis("off")
plt.imshow(ruidos_gaussiano, cmap='gray')
normalizada = np.zeros((256, 256))
normalizada = cv.normalize(ruidos_gaussiano, None, 0, 255, cv.NORM_MINMAX)
cv.imwrite('imagem_normalizada.jpg', normalizada)
plt.figure()
plt.axis("off")
plt.imshow(normalizada, cmap='gray')


img_norm = cv.imread('imagem_normalizada.jpg', 0)
print(img_norm)
median_blur_for_gaussian = cv.medianBlur(img_norm, 7)
plt.figure()
plt.axis("off")
plt.imshow(median_blur_for_gaussian, cmap='gray')






#plt.imsave('imagem_ruido_gaussiano.jpg', ruidos_gaussiano)


ruidos_sp = add_ruido_sal_e_pimenta(imagem_original)
plt.imsave('imagem_ruido_sal_pimenta.jpg', ruidos_sp)

#cv.imwrite('imagem_ruido_sal_pimenta.jpg', cv.cvtColor(ruidos_sp, cv.COLOR_BGR2RGB))



plt.show()
cv.waitKey()