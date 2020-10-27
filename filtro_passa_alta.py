import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from matplotlib import pylab
import os
from scipy import fftpack
from skimage.metrics import peak_signal_noise_ratio as psnr

dir_imagens_originais = './imagens_originais/'
lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_originais)

for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_original = imread(dir_imagens_originais +'/'+ nome_imagem, as_gray=True)

    freq = fp.fft2(imagem_original)
    (w, h) = freq.shape
    half_w, half_h = int(w/2), int(h/2)

    pylab.figure()

    #apply HPF
    #select all but the first lxl (low) frequencies
    for l in range(1, 30):
        freq1 = np.copy(freq)
        freq2 = fp.fftshift(freq1)

        spectro = (20 * np.log10(0.1 + freq2)).astype(int)
        freq2[half_w-l:half_w+l+1, half_h-l:half_h+l+1] = 0
        spectro_hf = (20 * np.log10(0.1+freq2)).astype(int)

        imagem_filtrada = np.clip(fp.ifft2(fftpack.ifftshift(freq2)).real, 0, 255)
    #print(img_as_ubyte(inversa))





        #pylab.subplot(6, 4, l)
        #pylab.title('Imagem Original')
        #pylab.imshow(imagem_original, cmap='gray')

        #pylab.subplot(6, 4, l)
        #pylab.title('Freq. Imagem Original')
        #pylab.imshow(spectro, cmap='gray')

        #pylab.subplot(6, 4, l)
        #pylab.title('Freq. Imagem Original')
        #pylab.imshow(spectro_hf, cmap='gray')

        pylab.subplot(6, 5, l)
        pylab.axis('off')
        pylab.title('F=' + str(l))
        pylab.imshow(imagem_filtrada, cmap='gray')

    pylab.subplots_adjust(wspace=0.1, hspace=0.5)
    pylab.show()
    pylab.close()


print('FIM FILTRO PASSA BAIXA')