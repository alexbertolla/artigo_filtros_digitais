import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte, img_as_float32
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy import fftpack
from matplotlib import pylab
import os
from os import path
import shutil


def calcular_psnr(imagem_ruidosa, imagem_filtrada):
    return round(psnr(imagem_ruidosa, imagem_filtrada), 2)

def gerar_spectro(transformata_imagem):
    shift_frq = fftpack.fftshift(transformata_imagem)
    return (20 * np.log10(0.1 + shift_frq)).real

def plotar_imagem(linha, coluna, posicao, imagem, titulo):
    pylab.subplot(linha, coluna, posicao)
    pylab.axis('off')
    pylab.title(titulo)
    pylab.imshow(imagem, cmap='gray')

def criar_diretorios_raizes(dir):
    shutil.rmtree(dir, ignore_errors=True)
    os.mkdir(dir)


#dir_destino = './imagens_originais_filtro_passa_baixa'
#shutil.rmtree(dir_destino, ignore_errors=True)
#os.mkdir(dir_destino)

lista_corte = (30, 35)

#dir_array = ([origem, destino, titulo])
dir_array = (
    ['./imagens_originais/', './imagens_originais_filtro_passa_baixa/', 'Imagens Originais/'],
    ['./imagens_ruido_gaussiano/', './imagens_ruido_gaussiano_filtro_passa_baixa/', 'Imagens Ruido Gaussiano/'],
    ['./imagens_ruido_sal_e_pimenta/', './imagens_ruido_sal_e_pimenta_filtro_passa_baixa/', 'Imagens Ruido Impulsivo/'],
    ['./imagens_ruido_spekle/', './imagens_ruido_spekle_filtro_passa_baixa/', 'Imagens Ruido Spekle/'],
)

for diretorio in dir_array:
    diretorio_origem = diretorio[0]
    diretorio_destino = diretorio[1]
    titulo = diretorio[2]
    criar_diretorios_raizes(diretorio_destino)
    for corte in lista_corte:
        criar_diretorios_raizes(diretorio_destino + str(corte))
        print('Trabalhando no diretório ', diretorio_destino + str(corte))
        lista_imagens = os.listdir(diretorio_origem)
        for nome_imagem in lista_imagens:
            imagem_original = img_as_float(imread(diretorio_origem + nome_imagem, as_gray=True))
            imagem_filtrada = img_as_float(np.zeros(imagem_original.shape))
            #print(imagem_filtrada.shape)
            #exit()

            linha, coluna = imagem_original.shape

            quadrante_1 = img_as_float(imagem_original[:int(linha / 2), :int(coluna / 2)])
            quadrante_2 = img_as_float(imagem_original[:int(linha / 2), int(coluna / 2):])
            quadrante_3 = img_as_float(imagem_original[int(linha / 2):, :int(coluna / 2)])
            quadrante_4 = img_as_float(imagem_original[int(linha / 2):, int(coluna / 2):])
            discrete_transform_imagem_original = fp.fft2(imagem_original)
            (w_io, h_io) = discrete_transform_imagem_original.shape
            half_w_io, half_h_io = int(w_io / 2), int(h_io / 2)
            spectro_imagem_original = gerar_spectro(discrete_transform_imagem_original)

            #discrete_transform_q1 = fp.fft2(quadrante_1)
            discrete_transform_q1 = fp.fft2(quadrante_1)
            (w_q1, h_q1) = discrete_transform_q1.shape
            half_w_q1, half_h_q1 = int(w_q1 / 2), int(h_q1 / 2)
            spectro_q1 = gerar_spectro(discrete_transform_q1)

            discrete_transform_q2 = fp.fft2(quadrante_2)
            (w_q2, h_q2) = discrete_transform_q2.shape
            half_w_q2, half_h_q2 = int(w_q2 / 2), int(h_q2 / 2)
            spectro_q2 = gerar_spectro(discrete_transform_q2)

            discrete_transform_q3 = fp.fft2(quadrante_3)
            (w_q3, h_q3) = discrete_transform_q3.shape
            half_w_q3, half_h_q3 = int(w_q3 / 2), int(h_q3 / 2)
            spectro_q3 = gerar_spectro(discrete_transform_q3)

            discrete_transform_q4 = fp.fft2(quadrante_4)
            (w_q4, h_q4) = discrete_transform_q4.shape
            half_w_q4, half_h_q4 = int(w_q4 / 2), int(h_q4 / 2)
            spectro_q4 = gerar_spectro(discrete_transform_q4)

            shift_frq_io = fftpack.fftshift(discrete_transform_imagem_original)
            shift_frq_low_io = np.copy(shift_frq_io)
            shift_frq_low_io[half_w_io - corte:half_w_io + corte + 1, half_h_io - corte:half_h_io + corte + 1] = 0
            shift_frq_io -= shift_frq_low_io
            imagem_original_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq_io)).real
            valor_psnr = calcular_psnr(imagem_original, imagem_original_filtrada)


            shift_frq_q1 = fftpack.fftshift(discrete_transform_q1)
            shift_frq_low_q1 = np.copy(shift_frq_q1)
            shift_frq_low_q1[half_w_q1 - corte:half_w_q1 + corte + 1, half_h_q1 - corte:half_h_q1 + corte + 1] = 0
            shift_frq_q1 -= shift_frq_low_q1
            q1_filtrado = img_as_float(fp.ifft2(fftpack.ifftshift(shift_frq_q1)).real)
            valor_psnr = calcular_psnr(quadrante_1, q1_filtrado)


            shift_frq_q2 = fftpack.fftshift(discrete_transform_q2)
            shift_frq_low_q2 = np.copy(shift_frq_q2)
            shift_frq_low_q2[half_w_q2 - corte:half_w_q2 + corte + 1, half_h_q2 - corte:half_h_q2 + corte + 1] = 0
            shift_frq_q2 -= shift_frq_low_q2
            q2_filtrado = img_as_float(fp.ifft2(fftpack.ifftshift(shift_frq_q2)).real)
            valor_psnr = calcular_psnr(quadrante_2, q2_filtrado)


            shift_frq_q3 = fftpack.fftshift(discrete_transform_q3)
            shift_frq_low_q3 = np.copy(shift_frq_q3)
            shift_frq_low_q3[half_w_q3 - corte:half_w_q3 + corte + 1, half_h_q3 - corte:half_h_q3 + corte + 1] = 0
            shift_frq_q3 -= shift_frq_low_q3
            q3_filtrado = img_as_float(fp.ifft2(fftpack.ifftshift(shift_frq_q3)).real)
            valor_psnr = calcular_psnr(quadrante_3, q3_filtrado)


            shift_frq_q4 = fftpack.fftshift(discrete_transform_q4)
            shift_frq_low_q4 = np.copy(shift_frq_q4)
            shift_frq_low_q4[half_w_q4 - corte:half_w_q4 + corte + 1, half_h_q4 - corte:half_h_q4 + corte + 1] = 0
            shift_frq_q4 -= shift_frq_low_q4
            q4_filtrado = img_as_float(fp.ifft2(fftpack.ifftshift(shift_frq_q4)).real)
            valor_psnr = calcular_psnr(quadrante_4, q4_filtrado)

            imagem_filtrada[:int(linha / 2), :int(coluna / 2)] = q1_filtrado
            imagem_filtrada[:int(linha / 2), int(coluna / 2):] = q2_filtrado
            imagem_filtrada[int(linha / 2):, :int(coluna / 2)] = q3_filtrado
            imagem_filtrada[int(linha / 2):, int(coluna / 2):] = q4_filtrado

            print(diretorio_destino + str(corte) + '/' + nome_imagem)

            #imagem_filtrada = np.array(imagem_filtrada, dtype='uint8')

            imsave(diretorio_destino + str(corte) + '/' + nome_imagem, imagem_filtrada)

            #pylab.figure()
            #pylab.suptitle(titulo + str(corte))
            #pylab.subplot(1, 3, 1)
            #pylab.axis('off')
            #pylab.imshow(imagem_original, cmap='gray')
            #pylab.subplot(1, 3, 2)
            #pylab.axis('off')
            #pylab.imshow(imagem_filtrada, cmap='gray')
            #pylab.subplot(1, 3, 3)
            #pylab.axis('off')
            #pylab.imshow(imagem_original_filtrada, cmap='gray')
            #pylab.show()

    #exit()




print('FIM FILTRO PASSA BAIXA')