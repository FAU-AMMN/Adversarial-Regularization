from ClassFiles.Framework import AdversarialRegulariser
from ClassFiles.networks import ConvNetClassifier
from ClassFiles.data_pips import BSDS
from ClassFiles.forward_models import Denoising


DATA_PATH = '/media/sriranjani/Data/masterThesis/DeepAdverserialRegulariser/data/BSR/BSDS500/data/images/'
SAVES_PATH = '/media/sriranjani/Data/masterThesis/DeepAdverserialRegulariser_torch/'


class Experiment1(AdversarialRegulariser):
    experiment_name = 'MediumNoise'
    noise_level = 0.01

    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = .3

    learning_rate = 0.0001
    step_size = .7
    total_steps_default = 50
    starting_point = 'Mini'

    def get_network(self, size, colors):
        return ConvNetClassifier(size=size, colors=colors)

    def unreg_mini(self, y, fbp):
        return self.update_pic(10, 1, y, fbp, 0)

    def get_Data_pip(self, data_path):
        return BSDS(data_path)

    def get_model(self, size):
        return Denoising(size=size)


experiment = Experiment1(DATA_PATH, SAVES_PATH)
experiment.find_good_lambda(32)
for k in range(7):
    experiment.train(100)
experiment.Network_Optimization_writer.close()
experiment.Reconstruction_Quality_writer.close()
experiment.log_optimization(32, 200, 0.7, .3)
experiment.log_optimization(32, 200, 0.7, .4)
experiment.log_optimization(32, 200, 0.7, .5)