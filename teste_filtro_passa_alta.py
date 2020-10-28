import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from matplotlib import pylab
import os
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio as psnr

dir_imagens_originais = './imagens_ruido_gaussiano/'

lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_originais)

for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_original = imread(dir_imagens_originais +'/'+ nome_imagem, as_gray=True)

    # Compute the 2-dimensional discrete Fourier Transform
    #freq = fp.fft2(imagem_original)
    discrete_transform = fp.fft2(imagem_original)
    (w, h) = discrete_transform.shape
    half_w, half_h = int(w/2), int(h/2)




    #apply HPF
    #select all but the first lxl (low) frequencies
    for l in (5, 10, 15, 20, 25, 30):
        #freq1 = np.copy(discrete_transform)

        # Shift the zero-frequency component to the center of the spectrum.
        shift_frq = fp.fftshift(discrete_transform)
        spectro_imagem_original = (20 * np.log10(0.1 + shift_frq)).real #.astype(int)




        shift_frq[half_w-l:half_w+l+1, half_h-l:half_h+l+1] = 0

        # fftpack.ifftshift()
        # The inverse of `fftshift`. Although identical for even-length `x`, the
        #     functions differ by one sample for odd-length `x`.
        imagem_alta_frequencia = np.clip(fp.ifft2(fftpack.ifftshift(shift_frq)).real, 0, 255)
        spectro_imagem_alta_frequencia = (20 * np.log10(0.1 + shift_frq)).real


        imagem_filtrada = (imagem_original + imagem_alta_frequencia)*255

        #imagem_filtrada = np.array(imagem_filtrada, dtype='uint8')



        pylab.figure()
        pylab.suptitle('Frequência de corte ' + str(l))

        pylab.subplot(1, 3, 1)
        pylab.axis('off')
        pylab.title('Imagem Original')
        pylab.imshow(imagem_original, cmap='gray')

        pylab.subplot(1, 3, 2)
        pylab.axis('off')
        pylab.title('Imagem Alta Frequência')
        pylab.imshow(imagem_alta_frequencia, cmap='gray')

        pylab.subplot(1, 3, 3)
        pylab.axis('off')
        pylab.title('Imagem Filtrada')
        pylab.imshow(imagem_filtrada, cmap='gray')

        #pylab.subplot(3, 3, 4)
        #pylab.axis('on')
        #pylab.title('Histograma Imagem Original')
        #pylab.hist(img_as_ubyte(imagem_original.flat), bins=256, range=(0, 255), color='black')

        #ylab.subplot(3, 3, 5)
        #pylab.axis('on')
        #pylab.title('Histograma Alta Frequência')
        #pylab.hist(img_as_ubyte(imagem_alta_frequencia.flat), bins=256, range=(0, 255), color='black')

        #pylab.subplot(3, 3, 6)
        #pylab.axis('on')
        #pylab.title('Histograma Imagem Filtrada')
        #pylab.hist(imagem_filtrada.flat, bins=256, range=(0, 255), color='black')


        #pylab.subplot(3, 3, 7)
        #pylab.axis('off')
        #pylab.title('Spectro Imagem Original')
        #pylab.imshow(spectro_imagem_original, cmap='gray')
        #pylab.hist(freq1.flat, bins=256, range=(-10, 10), color='black')

        #pylab.subplot(3, 3, 8)
        #pylab.axis('off')
        #pylab.title('Spectro Imagem Alta Frequência')
        #pylab.imshow(spectro_imagem_alta_frequencia, cmap='gray')
        # pylab.hist(freq1.flat, bins=256, range=(-10, 10), color='black')



        pylab.subplots_adjust(wspace=0.1, hspace=0.5)
        pylab.show()
        pylab.close()


print('FIM FILTRO PASSA BAIXA')