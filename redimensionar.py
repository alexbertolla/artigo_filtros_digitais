from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.transform import resize
import os
import shutil


dir_imagens_originais = 'imagens_originais'
dir_banco_imagens = 'banco_imagens'

shutil.rmtree(dir_banco_imagens, ignore_errors=True)
os.mkdir(dir_banco_imagens)

lista_imagens_originais = os.listdir(dir_imagens_originais)
total_imagem = len(lista_imagens_originais)
aux_total_imagem = 1
for nome_imagem in lista_imagens_originais:
    print(str(aux_total_imagem) + ' de ' + str(total_imagem) + ' imagens.')
    aux_total_imagem += 1
    imagem_original = img_as_float(imread(dir_imagens_originais + '/' + nome_imagem, as_gray=True))

    imagem_redimensionada = resize(imagem_original, (728, 728))
    imagem_redimensionada = img_as_ubyte(imagem_redimensionada)
    imsave(dir_banco_imagens + '/' + nome_imagem, imagem_redimensionada)
    #print(nome_imagem +' - '+ str(imagem_redimensionada.shape) + ' ' + str(imagem_redimensionada.dtype))
print('FIM REDIMENSIONAR')