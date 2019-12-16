import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
from skimage import color
from skimage.io import imread
from PIL import Image
nome_imagem_original = "imagem_original.jpg"


################### FUNÇÃO ADD RUIDO #######################
def add_ruido_branco(image):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob = 0.05
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 255
#            elif rdn > thres:
#                output[i][j] = 0
            else:
                output[i][j] = image[i][j]
    return output

from PIL import Image

def get_white_noise_image(width, height, imagem):
    #pil_map = Image.new("RGBA", (width, height), 255)
    pil_map = imagem
    print(pil_map)
    random_grid = map(lambda x: (
            int(random.random() * 255),
            int(random.random() * 255),
            int(random.random() * 255)
        ), [0] * width * height)
#    pil_map.putdata(random_grid)
    pil_map.putdata(list(random_grid))
    return pil_map


imagem_original = Image.open('imagem_original.jpg')
print(imagem_original)

ruido_branco = get_white_noise_image(256, 256, imagem_original)
plt.figure()
plt.imshow(ruido_branco)

plt.imsave('imagem_ruido_branco.jpg', ruido_branco)

imagem_ruido_branco = cv.imread('imagem_ruido_branco.jpg')
imagem_ruido_branco = cv.cvtColor(imagem_ruido_branco, cv.COLOR_BGR2RGB)
img_rest = cv.GaussianBlur(imagem_ruido_branco, (3, 3), 1)
plt.figure()
plt.imshow(img_rest)




plt.show()
