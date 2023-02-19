from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier
from ClassFiles.data_pips import BSDS, LUNA
from ClassFiles.forward_models import Denoising
import numpy as np
import time 

DATA_PATH = "/home/maniraman/Desktop/Ranjani/thesis/BSR/BSDS500/data/images/" #'/media/sriranjani/Data/masterThesis/DeepAdverserialRegulariser_torch/data/BSR/BSDS500/data/images/', '../Data/data/images/'
#DATA_PATH = '/home/maniraman/Desktop/Ranjani/thesis/LUNA/manifest-1674842977695/LIDC-IDRI/' #'../Data/luna/'
SAVES_PATH = '/home/maniraman/Desktop/Ranjani/thesis/git/Adversarial-Regularization/' #'/media/sriranjani/Data/masterThesis/git/Adversarial-Regularization/' #'../Saves/'

#Saves_clipped_Noise0.01
class Experiment1(AdversarialRegulariser):
    experiment_name = 'MediumNoise'
    data_clipping = "Unclipped_data"
    noise_level = 0.05

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = .3
    np.random.seed(0)
    
    learning_rate = 0.0001
    step_size = .7
    total_steps_default = 50
    starting_point = 'Mini'
    seed = 50

    def get_network(self, size, colors):
        return ConvNetClassifier(size, colors)
    
    def unreg_mini(self, y, fbp):
        return self.update_pic(10, 1, y, fbp, 0)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)

tm = time.time()

experiment = Experiment1(DATA_PATH, SAVES_PATH, "Noise0.05_goodlmb_adapPool")
experiment.noise_level = 0.05
experiment.data_clipping = 'Clipped_data'
lmb = experiment.find_good_lambda(32)
experiment.mu_default = lmb
"""for k in range(7):
    experiment.train(100)
experiment.Network_Optimization_writer.close()
experiment.Reconstruction_Quality_writer.close()"""
"""experiment.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=lmb)
experiment.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.4)
"""
experiment.log_optimization(batch_size=1, steps=200, step_s=0.7,mu=lmb)
experiment.log_optimization(batch_size=1, steps=200, step_s=0.7,mu=0.4)

duration = (time.time()-tm)/60
print("Time taken for experiment " + str(duration))