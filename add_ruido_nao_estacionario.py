import numpy as np
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import color
from matplotlib import pyplot as plt
import random
from skimage.util import random_noise

def add_ruido_gaussiano(imagem_original):
    sigma = 0.05
    imagem_ruido_gaussiano = random_noise(imagem_original, var=sigma)
    return imagem_ruido_gaussiano

def add_ruido_sal_e_pimenta(imagem_original):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob = 0.05
    imagem_ruido_sal_e_pimenta = np.zeros(imagem_original.shape)
    thres = 1 - prob

    for i in range(imagem_original.shape[0]):
        for j in range(imagem_original.shape[1]):
            rdn = random.random()
            if rdn < prob:
                imagem_ruido_sal_e_pimenta[i][j] = 0
            elif rdn > thres:
                imagem_ruido_sal_e_pimenta[i][j] = 1
            else:
                imagem_ruido_sal_e_pimenta[i][j] = imagem_original[i][j]
    return imagem_ruido_sal_e_pimenta

def add_ruido_spekle(imagem_original):
    imagem_ruido_spekle = np.copy(imagem_original)
    gauss = np.random.normal(0, 0.7, imagem_original.size)

    if len(imagem_original.shape) == 3:
        gauss = gauss.reshape(imagem_original.shape[0], imagem_original.shape[1], imagem_original.shape[2]).astype('uint8')
    else:
        gauss = gauss.reshape(imagem_original.shape[0], imagem_original.shape[1]).astype('uint8')
    imagem_ruido_spekle = img_as_ubyte(imagem_original) + img_as_ubyte(imagem_original) * gauss
    imagem_ruido_spekle = img_as_float(imagem_ruido_spekle)
    return imagem_ruido_spekle

lagarta = 'imagem_original.jpg'
lena = 'imagem_lena.png'

imagem_original = imread(lena, as_gray=True)
#imagem_original = imread(lagarta, as_gray=False)
imagem_original = img_as_float(imagem_original)

if len(imagem_original.shape) == 3:
    linha, coluna, _ = imagem_original.shape
else:
    linha, coluna = imagem_original.shape

imagem_ruidosa = np.copy(imagem_original)
imagem_ruidosa[:int(linha/2), :int(coluna/2)] = add_ruido_spekle(imagem_original[:int(linha/2), :int(coluna/2)])
imagem_ruidosa[:int(linha/2), int(coluna/2):] = add_ruido_sal_e_pimenta(imagem_original[:int(linha/2), int(coluna/2):])
imagem_ruidosa[int(linha/2):, :int(coluna/2)] = add_ruido_gaussiano(imagem_original[int(linha/2):, :int(coluna/2)])
#imagem_ruidosa[int(linha/2):, int(coluna/2):] = imagem_rgb[int(linha/2):, int(coluna/2):]

imagem_ruidosa = img_as_ubyte(imagem_ruidosa)
imsave('imagem_ruidosa.jpg', imagem_ruidosa)


plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Imagem Original')
plt.imshow(imagem_original, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Imagem Ruidosa')
plt.imshow(imagem_ruidosa, cmap='gray')

plt.subplot(2, 2, 3)
plt.title('Histograma Imagem Original')
plt.hist(imagem_original.flat, bins=256, range=(0, 1), color='black')

plt.subplot(2, 2, 4)
plt.title('Histograma Imagem Ruidosa')
plt.hist(imagem_ruidosa.flat, bins=256, range=(0, 1), color='black')

plt.show()


print('FIM')