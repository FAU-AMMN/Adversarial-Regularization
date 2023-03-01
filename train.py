#%%
from ClassFiles.data_pips import ellipses
import Adversarial_Regulariser
import numpy as np
import time 
import matplotlib.pyplot as plt


DATA_PATH = '../Data/data/images/'
SAVES_PATH = '../Saves/'
fname = 'fixed_data'

Adversarial_Regulariser.fix_seed(42)

#%%
experiment = Adversarial_Regulariser.Experiment1(DATA_PATH, SAVES_PATH, image_size = (128,128), exp_name= fname)
experiment.noise_level = 0.05

experiment.data_clipping = 'Clipped_data'
lmb = experiment.find_good_lambda(32)
experiment.mu_default = lmb

#%%
for k in range(7):
    experiment.train(100)
experiment.Network_Optimization_writer.close()
experiment.Reconstruction_Quality_writer.close()
# %%
experiment.log_optimization(batch_size=1, steps=200, step_s=0.1,mu=lmb)
# %%
