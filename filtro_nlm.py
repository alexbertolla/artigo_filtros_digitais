from skimage import img_as_float, img_as_ubyte
from skimage.io import imread, imsave
from skimage.restoration import estimate_sigma, denoise_nl_means
import os
import numpy as np

dir_imagens_ruido_gaussiano = 'imagens_ruido_gaussiano'
dir_imagens_ruido_spekle = 'imagens_ruido_spekle'
dir_imagens_ruido_sal_e_pimenta = 'imagens_ruido_sal_e_pimenta'

dir_imagens_ruido_gaussiano_filtro_nlm = 'imagens_ruido_gaussiano_filtro_nlm'
dir_imagens_ruido_spekle_filtro_nlm = 'imagens_ruido_spekle_filtro_nlm'
dir_imagens_ruido_sal_e_pimenta_filtro_nlm = 'imagens_ruido_sal_e_pimenta_filtro_nlm'

lista_imagens_ruido_gaussiano = os.listdir(dir_imagens_ruido_gaussiano)
for nome_imagem in lista_imagens_ruido_gaussiano:
    imagem_ruido_gaussiano = img_as_float(imread(dir_imagens_ruido_gaussiano + '/' + nome_imagem, as_gray=True))
    imagem_ruido_spekle = img_as_float(imread(dir_imagens_ruido_spekle + '/' + nome_imagem, as_gray=True))
    imagem_ruido_sal_e_pimenta = img_as_float(imread(dir_imagens_ruido_sal_e_pimenta + '/' + nome_imagem, as_gray=True))

    janela = 3
    sigma_gau_est = np.mean(estimate_sigma(img_as_ubyte(imagem_ruido_gaussiano), multichannel=False))
    imagem_ruido_gaussiano_filtrada = denoise_nl_means(imagem_ruido_gaussiano,
                                                       h=1.5 * sigma_gau_est,
                                                       fast_mode=True,
                                                       patch_size=janela,
                                                       patch_distance=3,
                                                       multichannel=False)

    sigma_spk_est = np.mean(estimate_sigma(img_as_ubyte(imagem_ruido_spekle), multichannel=False))
    imagem_ruido_spekle_filtrada = denoise_nl_means(imagem_ruido_spekle,
                                                       h=1.5 * sigma_spk_est,
                                                       fast_mode=True,
                                                       patch_size=janela,
                                                       patch_distance=3,
                                                       multichannel=False)

    sigma_sp_est = np.mean(estimate_sigma(img_as_ubyte(imagem_ruido_sal_e_pimenta), multichannel=False))
    imagem_ruido_sal_e_pimenta_filtrada = denoise_nl_means(imagem_ruido_sal_e_pimenta,
                                                    h=1.5 * sigma_sp_est,
                                                    fast_mode=True,
                                                    patch_size=janela,
                                                    patch_distance=3,
                                                    multichannel=False)

    imagem_ruido_gaussiano_filtrada = img_as_ubyte(imagem_ruido_gaussiano_filtrada)
    imagem_ruido_spekle_filtrada = img_as_ubyte(imagem_ruido_spekle_filtrada)
    imagem_ruido_sal_e_pimenta_filtrada = img_as_ubyte(imagem_ruido_sal_e_pimenta_filtrada)


    imsave(dir_imagens_ruido_gaussiano_filtro_nlm +'/'+ nome_imagem, imagem_ruido_gaussiano_filtrada)
    imsave(dir_imagens_ruido_spekle_filtro_nlm + '/' + nome_imagem, imagem_ruido_spekle_filtrada)
    imsave(dir_imagens_ruido_sal_e_pimenta_filtro_nlm + '/' + nome_imagem, imagem_ruido_sal_e_pimenta_filtrada)

    print(nome_imagem)

print('FIM FILTRO NON LOCAL MEANS')

