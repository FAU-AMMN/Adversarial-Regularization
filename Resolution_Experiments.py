
#%%
from ClassFiles.data_pips import ellipses
import Adversarial_Regulariser
import numpy as np
import time 
import matplotlib.pyplot as plt


DATA_PATH = '../Data/data/images/'
SAVES_PATH = '../Saves/'
fname = 'fixed_data'

resolutions = [(64,64),(128,128),(256,256), (512,512)]



#%%
for res in resolutions:
    Adversarial_Regulariser.fix_seed(0)
    experiment = Adversarial_Regulariser.Experiment1(DATA_PATH, SAVES_PATH, image_size = res, exp_name= fname)
    experiment.noise_level = 0.05

    experiment.data_clipping = 'Clipped_data'
    lmb = experiment.find_good_lambda(32)
    experiment.mu_default = lmb

    Adversarial_Regulariser.fix_seed(42)
    datapip = experiment.get_Data_pip(DATA_PATH, experiment.image_size)
    img = datapip.load_data()

    noisy_img = experiment.generate_training_data(batch_size = 1)[0]
    plt.figure()
    plt.imshow(img[:,:,0])
    plt.imshow(noisy_img[0,0])

    experiment.log_optimization(batch_size=1, steps=0, step_s=0,mu=lmb)
# %%
