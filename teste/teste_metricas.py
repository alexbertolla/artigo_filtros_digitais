from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from skimage.io import imread
from skimage import img_as_float, img_as_ubyte
import numpy as np
from scipy import stats
from skimage.util.dtype import dtype_range

imagem = img_as_float(imread('../banco_imagens/hz_1235134-PPT.jpg'))
#inicio snr
a = np.asanyarray(imagem)
m = a.mean(None)
sd = a.std(axis=None, ddof=0)
snr = np.where(sd == 0, 0, m/sd)
#fim snr

#inicio psnr
dmin, dmax = dtype_range[imagem.dtype.type]
true_min, true_max = np.min(imagem), np.max(imagem)
true_min = 2

data_range = 0
if true_min >= 0:
    data_range = dmax
else:
    data_range = dmax - dmin

mse = np.mean(imagem)**2
print(mse)

print(data_range)
#fim psnr

exit()

psnr = peak_signal_noise_ratio(imagem, np.zeros(imagem.shape))
mse = mean_squared_error(imagem, np.zeros(imagem.shape))

print(a)
print('m = ' + str(m))
print('sd = ' + str(sd))
print('snr = ' + str(snr))
print('psnr = ' + str(psnr))
print('mse = ' + str(mse))



print('FIM TESTE PSNR')