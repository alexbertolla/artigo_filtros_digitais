from matplotlib import pylab as plt
from skimage.io import imread, imsave
from skimage import img_as_float, img_as_ubyte
import os

dir_imagens_originais = '../imagens_originais'
dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'
dir_imagens_ruido_nao_estacionario = 'imagens_ruido_nao_estacionario'

dir_imagens_plots = 'imagens_plots'

dir_imagens_ruido_gaussiano_filtro_wiener = 'imagens_ruido_gaussiano_filtro_wiener'
dir_imagens_ruido_spekle_filtro_wiener = 'imagens_ruido_spekle_filtro_wiener'
dir_imagens_ruido_sal_e_pimenta_filtro_wiener = 'imagens_ruido_sal_e_pimenta_filtro_wiener'

lista_imagens_originais = os.listdir(dir_imagens_originais)
for nome_imagem in lista_imagens_originais:

    imagem_original = img_as_float(imread(dir_imagens_originais +'/' + nome_imagem, as_gray=True))
    imagem_ruido_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle = img_as_float(imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True))
    #imagem_ruido_nao_estacionario = img_as_float(imread(dir_imagens_ruido_nao_estacionario + '/' + nome_imagem, as_gray=True))


    plt.figure(figsize=(20, 20))
    plt.subplot(3, 5, 1)
    plt.axis('off')
    plt.title('Imagem Original')
    plt.imshow(imagem_original, cmap='gray')

    plt.subplot(3, 5, 6)
    plt.axis('on')
    plt.title('Histograma Original')
    plt.hist(img_as_ubyte(imagem_original.flat), bins=256, range=(0, 255), color='black')

    plt.subplot(3, 5, 2)
    plt.axis('off')
    plt.title('Imagem Ruído Gaussiano')
    plt.imshow(imagem_ruido_gaussiano, cmap='gray')

    plt.subplot(3, 5, 7)
    plt.axis('on')
    plt.title('Histograma Ruído Gaussiano')
    plt.hist(img_as_ubyte(imagem_ruido_gaussiano.flat), bins=256, range=(0, 255), color='black')

    plt.subplot(3, 5, 3)
    plt.axis('off')
    plt.title('Imagem Ruído Spekle')
    plt.imshow(imagem_ruido_spekle, cmap='gray')

    plt.subplot(3, 5, 8)
    plt.axis('on')
    plt.title('Histograma Ruído Spekle')
    plt.hist(img_as_ubyte(imagem_ruido_spekle.flat), bins=256, range=(0, 255), color='black')

    plt.subplot(3, 5, 4)
    plt.axis('off')
    plt.title('Imagem Ruído Sal e Pimenta')
    plt.imshow(imagem_ruido_sal_e_pimenta, cmap='gray')

    plt.subplot(3, 5, 9)
    plt.axis('on')
    plt.title('Histograma Ruído Sal e Pimenta')
    plt.hist(img_as_ubyte(imagem_ruido_sal_e_pimenta.flat), bins=256, range=(0, 255), color='black')

    plt.subplot(3, 5, 5)
    plt.axis('off')
    plt.title('Imagem Ruído Não Estanionário')
    #plt.imshow(imagem_ruido_nao_estacionario, cmap='gray')

    plt.subplot(3, 5, 10)
    plt.axis('on')
    plt.title('Histograma Ruído Não Estanionário')
    #plt.hist(img_as_ubyte(imagem_ruido_nao_estacionario.flat), bins=256, range=(0, 255), color='black')
    plt.show()

    #plt.savefig(dir_imagens_plots + '/plot_'+ nome_imagem + '.png')
    plt.close()
    print(nome_imagem)

    exit('FIM GERAR PLOTAGEM')


print('FIM GERAR PLOTAGEM')