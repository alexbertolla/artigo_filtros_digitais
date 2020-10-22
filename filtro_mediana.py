from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage import color
from skimage.filters import median
from skimage.morphology import disk
from matplotlib import pyplot as plt

imagem_original = imread('imagem_lena.png', as_gray=True)
imagem_ruidosa = imread('imagem_ruidosa.jpg', as_gray=True)
#imagem_ruidosa = imread('imagem_ruidosa.jpg', as_gray=False)

imagem_original = img_as_float(imagem_original)
imagem_ruidosa = img_as_float(imagem_ruidosa)

if len(imagem_ruidosa.shape) == 3:
    linha, coluna, _ = imagem_ruidosa.shape
else:
    linha, coluna = imagem_ruidosa.shape

janela = 3
imagem_filtrada = median(imagem_ruidosa, disk(janela), mode='constant', cval=0.0)
imagem_filtrada = img_as_ubyte(imagem_filtrada)

imagem_original_filtrada = median(imagem_original, disk(janela), mode='constant', cval=0.0)
imagem_original_filtrada = img_as_ubyte(imagem_original_filtrada)

imsave('imagem_original_filtro_mediana.jpg', imagem_original_filtrada)
imsave('imagem_ruidosa_filtro_mediana.jpg', imagem_filtrada)

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.title('Imagem Ruidosa')
plt.imshow(imagem_ruidosa, cmap='gray')

plt.subplot(2, 2, 2)
plt.title('Imagem Filtrada')
plt.imshow(imagem_filtrada, cmap='gray')



plt.show()
print('FIM FILTRO MEDIANA')

