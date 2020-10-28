import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from matplotlib import pylab
import os
from scipy import fftpack


dir_imagens_originais = './imagens_originais/'
lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_originais)

for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_original = imread(dir_imagens_originais +'/'+ nome_imagem, as_gray=True)

    discrete_transform = fp.fft2(imagem_original)
    (w, h) = discrete_transform.shape
    half_w, half_h = int(w / 2), int(h / 2)


    for l in (5, 10, 15, 20, 25, 30):
        #freq1 = np.copy(discrete_transform)
        shift_frq = fftpack.fftshift(discrete_transform)
        shift_frq_low = np.copy(shift_frq)
        shift_frq_low[half_w-l:half_w+l+1, half_h-l:half_h+l+1] = 0
        spectro_imagem_original = (20 * np.log10(0.1 + shift_frq)).real  # .astype(int)

        #select only the first lxl (low) frequencies
        shift_frq -= shift_frq_low

        imagem_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq)).real
        spectro_baixa_frequencia = (20 * np.log10(0.1 + shift_frq)).real  # .astype(int)

        pylab.figure()
        pylab.suptitle('Frequência de corte ' + str(l))

        pylab.subplot(3, 3, 1)
        pylab.axis('off')
        pylab.title('Imagem Original')
        pylab.imshow(imagem_original, cmap='gray')

        pylab.subplot(3, 3, 3)
        pylab.axis('off')
        pylab.title('Imagem Filtrada')
        pylab.imshow(imagem_filtrada, cmap='gray')

        pylab.subplot(3, 3, 4)
        pylab.axis('on')
        pylab.title('Histograma Imagem Original')
        pylab.hist(img_as_ubyte(imagem_original.flat), bins=256, range=(0, 255), color='black')

        pylab.subplot(3, 3, 6)
        pylab.axis('on')
        pylab.title('Histograma Imagem Filtrada')
        pylab.hist(imagem_filtrada.flat, bins=256, range=(0, 1), color='black')

        pylab.subplot(3, 3, 7)
        pylab.axis('off')
        pylab.title('Spectro Imagem Original')
        pylab.imshow(spectro_imagem_original, cmap='gray')

        pylab.subplot(3, 3, 8)
        pylab.axis('off')
        pylab.title('Spectro Imagem Baixa Frequência')
        pylab.imshow(spectro_baixa_frequencia, cmap='gray')

        pylab.subplots_adjust(wspace=0.1, hspace=0.5)
        pylab.show()
        pylab.close()

print('FIM FILTRO PASSA BAIXA')
