import numpy as np
from numpy import fft as fp
from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.metrics import peak_signal_noise_ratio as psnr
from matplotlib import pylab
import os
from scipy import fftpack

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


corte = (20, 25, 30, 35)

dir_array = (['./imagens_originais', 'Imagens Originais'],
             ['./imagens_ruido_gaussiano', 'Ruido Gaussiano'],
             ['./imagens_ruido_sal_e_pimenta', 'Ruido Impulsivo'],
             ['./imagens_ruido_spekle', 'Ruido Spekle']
             )

#dir_imagens_originais = './imagens_originais'
#dir_imagens_originais = './imagens_ruido_gaussiano'
#dir_imagens_originais = './imagens_ruido_sal_e_pimenta'
#dir_imagens_originais = './imagens_ruido_spekle'

for dir in dir_array:
    diretorio = dir[0]
    titulo  = dir[1]

    lista_imagens_ruido_gaussiano = os.listdir(diretorio)

    for nome_imagem in lista_imagens_ruido_gaussiano:
        plt_linha = len(corte)+1  #4
        plt_coluna = 5
        plt_posicao = 1

        imagem_original = img_as_float(imread(diretorio +'/'+ nome_imagem, as_gray=True))
        linha, coluna = imagem_original.shape

        quadrante_1 = imagem_original[:int(linha / 2), :int(coluna / 2)]
        quadrante_2 = imagem_original[:int(linha / 2), int(coluna / 2):]
        quadrante_3 = imagem_original[int(linha / 2):, :int(coluna / 2)]
        quadrante_4 = imagem_original[int(linha / 2):, int(coluna / 2):]

        discrete_transform_imagem_original = fp.fft2(imagem_original)
        (w_io, h_io) = discrete_transform_imagem_original.shape
        half_w_io, half_h_io = int(w_io / 2), int(h_io / 2)
        spectro_imagem_original = gerar_spectro(discrete_transform_imagem_original)

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


        pylab.figure()
        pylab.suptitle(titulo)

        plotar_imagem(plt_linha, plt_coluna, plt_posicao, imagem_original, 'Imagem Original')
        plt_posicao += 1

        plotar_imagem(plt_linha, plt_coluna, plt_posicao, quadrante_1, 'Q1')
        plt_posicao += 1

        plotar_imagem(plt_linha, plt_coluna, plt_posicao, quadrante_2, 'Q2')
        plt_posicao += 1

        plotar_imagem(plt_linha, plt_coluna, plt_posicao, quadrante_3, 'Q3')
        plt_posicao += 1

        plotar_imagem(plt_linha, plt_coluna, plt_posicao, quadrante_4, 'Q4')
        plt_posicao += 1

        for l in corte:
            valor_psnr = 0.00

            shift_frq_io = fftpack.fftshift(discrete_transform_imagem_original)
            shift_frq_low_io = np.copy(shift_frq_io)
            shift_frq_low_io[half_w_io - l:half_w_io + l + 1, half_h_io - l:half_h_io + l + 1] = 0
            shift_frq_io -= shift_frq_low_io
            imagem_original_filtrada = fp.ifft2(fftpack.ifftshift(shift_frq_io)).real
            valor_psnr = calcular_psnr(imagem_original, imagem_original_filtrada)

            plotar_imagem(plt_linha, plt_coluna, plt_posicao, imagem_original_filtrada, 'Corte ' + str(l) + ' PSNR ' + str(valor_psnr))
            plt_posicao += 1

            shift_frq_q1 = fftpack.fftshift(discrete_transform_q1)
            shift_frq_low_q1 = np.copy(shift_frq_q1)
            shift_frq_low_q1[half_w_q1 - l:half_w_q1 + l + 1, half_h_q1 - l:half_h_q1 + l + 1] = 0
            shift_frq_q1 -= shift_frq_low_q1
            q1_filtrado = fp.ifft2(fftpack.ifftshift(shift_frq_q1)).real
            valor_psnr = calcular_psnr(quadrante_1, q1_filtrado)

            plotar_imagem(plt_linha, plt_coluna, plt_posicao, q1_filtrado, 'Corte ' + str(l) + ' PSNR ' + str(valor_psnr))
            plt_posicao += 1

            shift_frq_q2 = fftpack.fftshift(discrete_transform_q2)
            shift_frq_low_q2 = np.copy(shift_frq_q2)
            shift_frq_low_q2[half_w_q2 - l:half_w_q2 + l + 1, half_h_q2 - l:half_h_q2 + l + 1] = 0
            shift_frq_q2 -= shift_frq_low_q2
            q2_filtrado = fp.ifft2(fftpack.ifftshift(shift_frq_q2)).real
            valor_psnr = calcular_psnr(quadrante_2, q2_filtrado)

            plotar_imagem(plt_linha, plt_coluna, plt_posicao, q2_filtrado, 'Corte ' + str(l) + ' PSNR ' + str(valor_psnr))
            plt_posicao += 1

            shift_frq_q3 = fftpack.fftshift(discrete_transform_q3)
            shift_frq_low_q3 = np.copy(shift_frq_q3)
            shift_frq_low_q3[half_w_q3 - l:half_w_q3 + l + 1, half_h_q3 - l:half_h_q3 + l + 1] = 0
            shift_frq_q3 -= shift_frq_low_q3
            q3_filtrado = fp.ifft2(fftpack.ifftshift(shift_frq_q3)).real
            valor_psnr = calcular_psnr(quadrante_3, q3_filtrado)

            plotar_imagem(plt_linha, plt_coluna, plt_posicao, q3_filtrado, 'Corte ' + str(l) + ' PSNR ' + str(valor_psnr))
            plt_posicao += 1

            shift_frq_q4 = fftpack.fftshift(discrete_transform_q4)
            shift_frq_low_q4 = np.copy(shift_frq_q4)
            shift_frq_low_q4[half_w_q4 - l:half_w_q4 + l + 1, half_h_q4 - l:half_h_q4 + l + 1] = 0
            shift_frq_q4 -= shift_frq_low_q4
            q4_filtrado = fp.ifft2(fftpack.ifftshift(shift_frq_q4)).real
            valor_psnr = calcular_psnr(quadrante_4, q4_filtrado)

            plotar_imagem(plt_linha, plt_coluna, plt_posicao, q4_filtrado, 'Corte ' + str(l) + ' PSNR ' + str(valor_psnr))
            plt_posicao += 1


            #spectro_baixa_frequencia = (20 * np.log10(0.1 + shift_frq)).real  # .astype(int)


        #pylab.subplots_adjust(wspace=0.1, hspace=0.5)
        pylab.show()
        pylab.close()

print('FIM FILTRO PASSA BAIXA')
