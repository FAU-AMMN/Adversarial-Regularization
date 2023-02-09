from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier, FNO2d
from ClassFiles.data_pips import BSDS
from ClassFiles.forward_models import Denoising
import numpy as np
import time 
import torch

DATA_PATH = '/media/sriranjani/Data/masterThesis/DeepAdverserialRegulariser/data/BSR/BSDS500/data/images/' #'../Data/data/images/'
SAVES_PATH = '/media/sriranjani/Data/masterThesis/DeepAdverserialRegulariser_torch/' #'../Saves/'


#Saves_clipped_Noise0.01
class Experiment1(AdversarialRegulariser):
    experiment_name = 'MediumNoise'
    data_clipping = "Unclipped_data"
    noise_level = 0.07

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
        return ConvNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        check = torch.square(torch.tensor(y) -torch.tensor(fbp))
        return self.update_pic(10, 1, y, fbp, 0)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)

"""
experiment = Experiment1(DATA_PATH, SAVES_PATH, "Noise0.05_unclipped_goodlmb", train_model= False)
#experiment.data_clipping = 'Clipped_data'
lmb = experiment.find_good_lambda(32)
experiment.mu_default = lmb
experiment.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=lmb)

"""
tm = time.time()

experiment = Experiment1(DATA_PATH, SAVES_PATH, "Noise0.07_Clipped_goodlmb")
experiment.data_clipping = 'Clipped_data'
lmb = experiment.find_good_lambda(32)
experiment.mu_default = lmb
for k in range(7):
    experiment.train(100)
experiment.Network_Optimization_writer.close()
experiment.Reconstruction_Quality_writer.close()
experiment.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=lmb)
experiment.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.4)
experiment.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.5)

duration = (time.time()-tm)/60
print("Time taken for experiment" +str(duration))

"""
experiment1 = Experiment1(DATA_PATH, SAVES_PATH, "Noise0.15_Clipped_defaultlmb")
experiment1.data_clipping = 'Clipped_data'
lmb = experiment1.find_good_lambda(32)
for k in range(7):
    experiment1.train(100)
experiment1.Network_Optimization_writer.close()
experiment1.Reconstruction_Quality_writer.close()
experiment1.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=lmb)
experiment1.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.4)
experiment1.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.5)


experiment2 = Experiment1(DATA_PATH, SAVES_PATH, "Noise0.15_unclipped_goodlmb")
lmb = experiment2.find_good_lambda(32)
experiment2.mu_default = lmb
for k in range(7):
    experiment2.train(100)
experiment2.Network_Optimization_writer.close()
experiment2.Reconstruction_Quality_writer.close()
experiment2.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=lmb)
experiment2.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.4)
experiment2.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.5)
"""

"""experiment3 = Experiment1(DATA_PATH, SAVES_PATH, "Noise0.1_unclipped_defaultlmb")
lmb = experiment3.find_good_lambda(32)
for k in range(7):
    experiment3.train(100)
experiment3.Network_Optimization_writer.close()
experiment3.Reconstruction_Quality_writer.close()
experiment3.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=lmb)
experiment3.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.4)
experiment3.log_optimization(batch_size=32, steps=200, step_s=0.7,mu=0.5)"""

"""duration = (time.time()-tm)/60
print("Time taken for all experiment " +str(duration))"""