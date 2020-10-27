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

    freq = fp.fft2(imagem_original)
    (w, h) = freq.shape
    half_w, half_h = int(w / 2), int(h / 2)

    pylab.figure()
    for l in range(1, 30):
        freq1 = np.copy(freq)
        freq2 = fftpack.fftshift(freq1)
        freq2_low = np.copy(freq2)
        freq2_low[half_w-l:half_w+l+1, half_h-l:half_h+l+1] = 0

        #select only the first lxl (low) frequencies
        freq2 -= freq2_low

        imagem_filtrada = fp.ifft2(fftpack.ifftshift(freq2)).real


        pylab.subplot(6, 5, l)
        pylab.axis('off')
        pylab.title('F=' + str(l))
        pylab.imshow(imagem_filtrada, cmap='gray')

    pylab.subplots_adjust(wspace=0.1, hspace=0.5)
    pylab.show()
    pylab.close()

print('FIM FILTRO PASSA BAIXA')
