
#%%
from ClassFiles.data_pips import ellipses
import Adversarial_Regulariser
import numpy as np
import time 
import matplotlib.pyplot as plt

#%%
import sys, os
sys.path.append(os.path.abspath('../FourierImaging/'))

import fourierimaging as fi


#%%

DATA_PATH = '../Data/data/images/'
SAVES_PATH = '../Saves/'
fname = 'fno_nostride'

resolutions = [(64,64),(128,128), (256,256)]



#%%
## Test generation of phantoms for different resolutions
## Note: PSNR is discretization invariant, SSIM is not, l2 not by factor height*width
for res in resolutions:
    Adversarial_Regulariser.fix_seed(42)
    experiment = Adversarial_Regulariser.Experiment1(DATA_PATH, SAVES_PATH, image_size = res, exp_name= fname)
    experiment.noise_level = 0.05*res[0]/64

    experiment.data_clipping = 'Clipped_data'
    lmb = experiment.find_good_lambda(32)
    experiment.mu_default = lmb

    Adversarial_Regulariser.fix_seed(42)
    datapip = experiment.get_Data_pip(DATA_PATH, experiment.image_size)
    img = datapip.load_data()

    noisy_img = experiment.generate_training_data(batch_size = 1)[0]
    plt.figure()
    plt.imshow(img[:,:,0])
    plt.figure()
    plt.imshow(noisy_img[0,0])

    #experiment.log_optimization(batch_size=1, steps=200, step_s=0.01,mu=lmb)
# %%

net = experiment.get_network(size=None, colors=1)
fkernel = net.conv2.weight
kernel = fi.modules.spectral_to_spatial(fkernel, [128,128], odd = True, conv_like_cnn = True)
plt.figure()
plt.imshow(kernel[0,0,60:69,60:69].detach().numpy())
print(np.log(kernel[0,0,60:69,60:69].detach().numpy()))
# %%
