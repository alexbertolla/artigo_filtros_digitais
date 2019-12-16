import cv2 as cv
from skimage.metrics import structural_similarity as ssim
from add_ruido_sal_e_pimenta import add_ruido_sal_e_pimenta

nome_imagem_original = "imagem_original.jpg"
imagem_original = cv.imread(nome_imagem_original)
imagem_original = cv.cvtColor(imagem_original, cv.COLOR_BGR2GRAY)

imagem_2 = add_ruido_sal_e_pimenta(imagem_original)

var_ssim = ssim(imagem_original, imagem_2)
print(var_ssim)
