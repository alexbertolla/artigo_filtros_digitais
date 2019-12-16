import numpy as np
from PIL import ImageFilter as imfilter
from PIL import Image

#from skimage import color, viewer, exposure, img_as_float, data

from matplotlib import pyplot as plt

from add_ruido_gaussiano import add_ruido_gaussiano

#pylab.figure(figsize=(20, 35))

im = Image.open("imagem_original.jpg")
#print(im)
pix = np.array(im)
#print(pix)
pil_img = Image.fromarray(pix)
#print(pil_img)
#plt.figure()
#plt.axis("off")
#plt.imshow(im)

ruidos_gaussiano = add_ruido_gaussiano(pix)
#pil_img_g = Image.fromarray(ruidos_gaussiano)
#ruidos_sp = add_ruido_sal_e_pimenta.add_ruido_sal_e_pimenta(pix)
#pil_img_sp = Image.fromarray(ruidos_sp)

plt.figure()
plt.axis("off")
plt.imshow(ruidos_gaussiano)
print(ruidos_gaussiano)

#plt.figure()
#plt.axis("off")
#plt.imshow(ruidos_sp)

im1 = im.filter(imfilter.MedianFilter(size=3))

#plt.figure()
#plt.axis("off")
#plt.imshow(im1)

plt.show()

print('FIM')