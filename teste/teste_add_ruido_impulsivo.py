from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
import numpy as np
from matplotlib import pylab
import random

def adicionar_ruido_impulsivo(imagem, intensidade_ruido):
    #imagem = img_as_ubyte(imagem)
    #imagem_ruidosa = np.zeros(imagem.shape, np.uint8)
    imagem_ruidosa = np.zeros(imagem.shape, np.float)
    thres = 1 - intensidade_ruido
    print(thres)
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            rdn = random.random()
            if rdn < intensidade_ruido:
                imagem_ruidosa[i][j] = 0
            elif rdn > thres:
                imagem_ruidosa[i][j] = 1 #255
            else:
                imagem_ruidosa[i][j] = imagem[i][j]
    return imagem_ruidosa

lista_insensidades_ruido = [0.00, 0.05, 0.10, 0.15]
dir_imagens = '../banco_imagens/hz_imagem_64.jpg'
imagem_original = img_as_float(imread(dir_imagens, as_gray=True))

for intensidade in lista_insensidades_ruido:
    imagem_ruido = adicionar_ruido_impulsivo(imagem_original, intensidade)
    pylab.title(intensidade)
    pylab.imshow(img_as_ubyte(imagem_ruido), cmap='gray')
    pylab.show()

print('FIM TESTE ADD RUÍDO IMPULSIVO')