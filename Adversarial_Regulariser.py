from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier_nostride, Spectral_withResize, Spectral_FromTrainedConv
from ClassFiles.data_pips import BSDS, ellipses
from ClassFiles.forward_models import Denoising
import numpy as np
import time 
import random
import torch

def fix_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

fix_seed()
DATA_PATH = "/home/maniraman/Desktop/Ranjani/thesis/BSR/BSDS500/data/images/" #'/media/sriranjani/Data/masterThesis/DeepAdverserialRegulariser_torch/data/BSR/BSDS500/data/images/', '../Data/data/images/'
SAVES_PATH = '/home/maniraman/Desktop/Ranjani/thesis/git/Adversarial-Regularization/' #'/media/sriranjani/Data/masterThesis/git/Adversarial-Regularization/' #'../Saves/'


#Saves_clipped_Noise0.01
class Experiment1(AdversarialRegulariser):
    experiment_name = 'MediumNoise'
    data_clipping = "Unclipped_data"
    noise_level = 0.01

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = .3
    
    learning_rate = 0.0001
    step_size = .7
    total_steps_default = 50
    starting_point = 'Mini'

    def get_network(self, size, colors):
        return Spectral_withResize(size, colors)
    
    def unreg_mini(self, y, fbp):
        return self.update_pic(10, 1, y, fbp, 0)

    def get_Data_pip(self, data_path, image_size = None):
        image_size = (240, 240) #change for different dimension during evaluation
        return BSDS(data_path, image_size=image_size)

    def get_model(self, size):
        return  Denoising(size=size)

tm = time.time()

train_model = False #set false when loading weights for testing

experiment = Experiment1(DATA_PATH, SAVES_PATH, exp_name="Noise0.05_Spectral_withResize_200",
                        model_input_size=(200,200), #image size during training
                        conv_to_spectral=False,  #False while training CNN or FNO 
                        train_model=train_model) 

experiment.noise_level = 0.05
experiment.data_clipping = 'Clipped_data'
lmb = experiment.find_good_lambda(32)
experiment.mu_default = lmb

if train_model:
    for k in range(7):
        experiment.train(100)
    experiment.Network_Optimization_writer.close()
    experiment.Reconstruction_Quality_writer.close()
    experiment.Network_Optimization_csv.close()
    experiment.Reconstruction_Quality_csv.close()

experiment.log_optimization(batch_size=8, steps=200, step_s=0.01,mu=lmb, logOpti=False) #logOpti=False will resize BSDS image

duration = (time.time()-tm)/60
print("Time taken for experiment " + str(duration))