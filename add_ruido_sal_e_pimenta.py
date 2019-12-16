import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
nome_imagem_original = "imagem_original.jpg"


################### FUNÇÃO ADD RUIDO #######################
def add_ruido_sal_e_pimenta(image):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    prob = 0.1
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


################# RUIDO SAL E PIMENTA #################

#imagem_original = cv.imread(nome_imagem_original, cv.COLOR_BGR2RGB)

#imagem_cinza = cv.cvtColor(imagem_original, cv.COLOR_RGB2GRAY)
#plt.figure()
#plt.axis("off")
#plt.imshow(imagem_cinza, cmap="gray")

#ruido1 = sp_noise(imagem_cinza, 0.10)

#plt.figure()
#plt.axis("off")

#plt.imshow(ruido1, cmap="gray")



#plt.show()

#cv.waitKey()