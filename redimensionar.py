from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import resize
import os


dir_imagens = 'imagens_originais'

lista_imagens_originais = os.listdir(dir_imagens)
for nome_imagem in lista_imagens_originais:
    imagem_original = imread(dir_imagens + '/' + nome_imagem)
    imagem_redimensionada = resize(imagem_original, (728, 728))
    imagem_redimensionada = img_as_ubyte(imagem_redimensionada)
    imsave(dir_imagens + '/' + nome_imagem, imagem_redimensionada)
    print(dir_imagens + '/' + nome_imagem +' - '+ str(imagem_redimensionada.shape))
print('FIM ADD RUIDO RESIZE')