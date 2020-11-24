from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import resize
import os


dir_imagens_originais = 'imagens_originais'
dir_banco_imagens = 'banco_imagens'

lista_imagens_originais = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens_originais:
    imagem_original = img_as_float(imread(dir_imagens_originais + '/' + nome_imagem, as_gray=True))

    imagem_redimensionada = resize(imagem_original, (728, 728))
    imagem_redimensionada = img_as_ubyte(imagem_redimensionada)
    imsave(dir_banco_imagens + '/' + nome_imagem, imagem_redimensionada)
    print(nome_imagem +' - '+ str(imagem_redimensionada.shape) + ' ' + str(imagem_redimensionada.dtype))
print('FIM ADD RUIDO RESIZE')