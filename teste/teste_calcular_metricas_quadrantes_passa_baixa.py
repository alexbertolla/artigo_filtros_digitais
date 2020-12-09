import numpy as np
from skimage.io import imread
from skimage import img_as_ubyte, img_as_float
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import os
def calcular_mse(img1, img2):
    return round(mse(img1, img2), 2)


def calcular_psnr(img1, img2):
    return round(psnr(img1, img2), 2)



caminho_imagem_original = '../banco_imagens/'
caminho_imagem_ruidosa = '../imagens_ruido_gaussiano/'
caminho_filtro_passa_baixa = '../imagens_filtro_passa_baixa/'

lista_q1 = []
lista_q2 = []
lista_q3 = []
lista_q4 = []

lista_melhores_q1 = []
lista_melhores_q2 = []
lista_melhores_q3 = []
lista_melhores_q4 = []

lista_porcentagem_corte_passa_baixa = os.listdir(caminho_filtro_passa_baixa)
for corte in lista_porcentagem_corte_passa_baixa:
    dir_imagens_passa_baixa = caminho_filtro_passa_baixa + corte + '/'
    sublista_q1 = []
    sublista_q2 = []
    sublista_q3 = []
    sublista_q4 = []

    lista_imagens_passa_baixa = os.listdir(dir_imagens_passa_baixa)
    for nome_imagem in lista_imagens_passa_baixa:
        imagem_passa_baixa = img_as_float(imread(dir_imagens_passa_baixa + nome_imagem, as_gray=True))
        imagem_ruido = img_as_float(imread(caminho_imagem_ruidosa + nome_imagem, as_gray=True))
        imagem_original = img_as_float(imread(caminho_imagem_original + nome_imagem, as_gray=True))
        l, c = imagem_ruido.shape

        q1_ruidoso = imagem_ruido[:int(l / 2), :int(c / 2)]
        q2_ruidoso = imagem_ruido[:int(l / 2), int(c / 2):]
        q3_ruidoso = imagem_ruido[int(l / 2), :int(c / 2)]
        q4_ruidoso = imagem_ruido[int(l / 2), int(c / 2):]

        q1_pb = imagem_passa_baixa[:int(l / 2), :int(c / 2)]
        q2_pb = imagem_passa_baixa[:int(l / 2), int(c / 2):]
        q3_pb = imagem_passa_baixa[int(l / 2), :int(c / 2)]
        q4_pb = imagem_passa_baixa[int(l / 2), int(c / 2):]

        q1_psnr = calcular_psnr(q1_ruidoso, q1_pb)
        print('PSNR imagem_original/imagem_filtrada: ' + str(calcular_psnr(imagem_original, imagem_passa_baixa)))
        print('PSNR imagem_filtrada/imagem_original: ' + str(calcular_psnr(imagem_passa_baixa, imagem_original)))
        print()
        print('PSNR imagem_filtrada/imagem_ruido: ' + str(calcular_psnr(imagem_passa_baixa, imagem_ruido)))
        print('PSNR imagem_ruido/imagem_filtrada: ' + str(calcular_psnr(imagem_ruido, imagem_passa_baixa)))
        exit()
        q2_psnr = calcular_psnr(q2_ruidoso, q2_pb)
        q3_psnr = calcular_psnr(q3_ruidoso, q3_pb)
        q4_psnr = calcular_psnr(q4_ruidoso, q4_pb)

        sublista_q1.append(q1_psnr)
        sublista_q2.append(q2_psnr)
        sublista_q3.append(q3_psnr)
        sublista_q4.append(q4_psnr)

    lista_q1.append([corte, sublista_q1])
    lista_q2.append([corte, sublista_q2])
    lista_q3.append([corte, sublista_q3])
    lista_q4.append([corte, sublista_q4])



    q1_max1_psnr = np.max(sublista_q1)
    q2_max1_psnr = np.max(sublista_q2)
    q3_max1_psnr = np.max(sublista_q3)
    q4_max1_psnr = np.max(sublista_q4)

    sublista_q1.remove(q1_max1_psnr)
    sublista_q2.remove(q2_max1_psnr)
    sublista_q3.remove(q3_max1_psnr)
    sublista_q4.remove(q4_max1_psnr)


    q1_max2_psnr = np.max(sublista_q1)
    q2_max2_psnr = np.max(sublista_q2)
    q3_max2_psnr = np.max(sublista_q3)
    q4_max2_psnr = np.max(sublista_q4)



    sublista_q1.remove(q1_max2_psnr)
    sublista_q2.remove(q2_max2_psnr)
    sublista_q3.remove(q3_max2_psnr)
    sublista_q4.remove(q4_max2_psnr)

    q1_max3_psnr = np.max(sublista_q1)
    q2_max3_psnr = np.max(sublista_q2)
    q3_max3_psnr = np.max(sublista_q3)
    q4_max3_psnr = np.max(sublista_q4)

    sublista_q1.remove(q1_max3_psnr)
    sublista_q2.remove(q2_max3_psnr)
    sublista_q3.remove(q3_max3_psnr)
    sublista_q4.remove(q4_max3_psnr)

    lista_melhores_q1.append([corte, q1_max1_psnr])

    print('Filtro passa baixa, porcentagem de corte: ' + str(corte))
    print('Melhores PSNRs Q1: ' + str(q1_max1_psnr) + ', ' + str(q1_max2_psnr) + ', ' + str(q1_max3_psnr))
    print('Melhores PSNRs Q2: ' + str(q2_max1_psnr) + ', ' + str(q2_max2_psnr) + ', ' + str(q2_max3_psnr))
    print('Melhores PSNRs Q3: ' + str(q3_max1_psnr) + ', ' + str(q3_max2_psnr) + ', ' + str(q3_max3_psnr))
    print('Melhores PSNRs Q4: ' + str(q4_max1_psnr) + ', ' + str(q4_max2_psnr) + ', ' + str(q4_max3_psnr))

    print()



print('FIM TESTE CALCULAR METRICAS QUADRANTES PASSA BAIXA')