import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import random
from skimage.util import random_noise as rn


################### FUNÇÃO ADD RUIDO #######################
def add_ruido_gaussiano(image):
    #sigmas = [0.01, 0.05, 0.1, 0.15]
    #imagens = []
    #for s in sigmas:
    #    noisy = rn(image, var=s)
    #    imagens.append(noisy)
    #return imagens
    sigma = 0.05
    return rn(image, var=sigma)

