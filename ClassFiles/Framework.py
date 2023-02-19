import nntplib
import numpy as np
import os
import odl
from abc import ABC, abstractmethod
from ClassFiles import util as ut
import torch
import torch.nn as nn
from torch.utils.tensorboard.writer import SummaryWriter

class GenericFramework(ABC):
    model_name = 'no_model'
    experiment_name = 'default_experiment'
    data_clipping = 'Unclipped_data'

    # set the noise level used for experiments
    noise_level = 0.02

    @abstractmethod
    def get_network(self, size, colors):
        # returns an object of the network class. Used to set the network used
        pass

    @abstractmethod
    def get_Data_pip(self, path):
        # returns an object of the data_pip class.
        pass

    @abstractmethod
    def get_model(self, size):
        # Returns an object of the forward_model class.
        pass

    def __init__(self, data_path, saves_path, exp_name=None):
        self.data_pip = self.get_Data_pip(data_path)
        self.colors = self.data_pip.colors
        self.image_size = self.data_pip.image_size
        self.network = self.get_network(self.image_size, self.colors)
        self.model = self.get_model(self.image_size)
        self.image_space = self.model.get_image_size()
        self.measurement_space = self.model.get_measurement_size()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        if exp_name != None:
            self.experiment_name = exp_name
        # finding the correct path for saving models
        self.path = saves_path+'Saves/{}/{}/{}/{}/'.format(self.model.name, self.data_pip.name,
                                                           self.model_name, self.experiment_name)
        # generate needed folder structure
        ut.create_single_folder(self.path+'Data')
        ut.create_single_folder(self.path + 'Logs')

    def generate_training_data(self, batch_size, training_data=True, logOpti=False):
        # method to generate training data given the current model type
        y = np.empty(
            (batch_size,  self.colors, self.measurement_space[0], self.measurement_space[1]), dtype='float32')
        x_true = np.empty(
            (batch_size,  self.colors, self.image_space[0], self.image_space[1]), dtype='float32')
        fbp = np.empty(
            (batch_size, self.colors, self.image_space[0], self.image_space[1]), dtype='float32')

        for i in range(batch_size):
            if training_data:
                
                image = self.data_pip.load_data(training_data=True)
                np.random.seed(self.seed) ; noise = np.random.normal(size=(self.measurement_space[0],
                                               self.measurement_space[1],
                                               self.colors))
            else:
                if logOpti == False:
                    image = self.data_pip.load_data(training_data=False, logOpti=False)
                    np.random.seed(self.seed) ; noise = np.random.normal(size=(self.measurement_space[0],
                                                    self.measurement_space[1],
                                                    self.colors))
                else:
                    image = self.data_pip.load_data(training_data=False, logOpti=True)
                    y = np.empty((batch_size,  self.colors, image.shape[0], image.shape[1]), dtype='float32')
                    x_true = np.empty((batch_size,  self.colors, image.shape[0], image.shape[1]), dtype='float32')
                    fbp = np.empty((batch_size, self.colors, image.shape[0], image.shape[1]), dtype='float32')
                    np.random.seed(self.seed) ; noise = np.random.normal(size=(image.shape[0], image.shape[1],self.colors))
            data = self.model.forward_operator(image)

            # add white Gaussian noise
            noisy_data = data + self.noise_level*noise
            #noisy_data = data
            if self.data_clipping == "Clipped_data":
                noisy_data = np.clip(noisy_data, a_min = 0, a_max = 1)                                                                                    
            fbp[i, ...] = np.transpose(self.model.inverse(noisy_data), axes=[2,0,1])
            x_true[i, ...] = np.transpose(image, axes=[2,0,1])
            y[i, ...] = np.transpose(noisy_data, axes=[2,0,1])
        return y, x_true, fbp

    def save(self, global_step):
        torch.save({
            'global_step': global_step,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, self.path+'Data/model-'+str(global_step)+'.pth')
        print('Progress saved')
    
    def load(self):
        dirFiles = os.listdir(self.path+'Data/')
        if len(dirFiles) != 0:
            #dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
            dirFiles.sort()
            checkpoint = torch.load(self.path+'Data/'+dirFiles[-1])
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Save restored')
        else:
            print('No save found')


    @abstractmethod
    def evaluate(self, guess, measurement):
        # apply the model to data
        pass

class AdversarialRegulariser(GenericFramework):
    model_name = 'Adversarial_Regulariser'
    # the absolut noise level
    batch_size = 16
    # relation between L2 error and regulariser
    # 0 corresponds to pure L2 loss, infty to pure adversarial loss
    mu_default = 1.5
    # weight on gradient norm regulariser for wasserstein network
    lmb = 20
    # learning rate for Adams
    learning_rate = 0.0001
    # default step size for picture optimization
    step_size = 1
    # the amount of steps of gradient descent taken on loss functional
    total_steps_default = 30
    # default sampling pattern
    starting_point = 'Mini'

    def set_total_steps(self, steps):
        self.total_steps = steps

    # sets up the network architecture
    def __init__(self, data_path, saves_path, exp_name=None, train_model=True):
        # call superclass init 
        super(AdversarialRegulariser, self).__init__(data_path, saves_path, exp_name)
        self.total_steps = self.total_steps_default
        self.optimizer = torch.optim.RMSprop(self.network.parameters(), lr=self.learning_rate)
        if train_model == True:
            self.Network_Optimization_writer = SummaryWriter(log_dir=self.path + 'Logs/Network_Optimization/', comment='Network_Optimization')
            self.Reconstruction_Quality_writer = SummaryWriter(log_dir=self.path + 'Logs/Network_Optimization/', comment='Reconstruction_Quality')

        self.load()
    #self.update_pic(1, step_s, y, guess, mu)
    def update_pic(self, steps, stepsize, measurement, guess, mu):
        
        for k in range(steps):
            gradient, fullError = self.calculate_pic_grad(reconstruction=guess, data_term=measurement, mu= mu)
            guess = guess - stepsize * gradient
        return guess, fullError

    def unreg_mini(self, y, fbp):
        check = torch.square(torch.tensor(y) -torch.tensor(fbp))
        return self.update_pic(10, 0.1, y, fbp, 0)

    def log_minimum(self):
        y, x_true, fbp = self.generate_training_data(
            self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess, fullError = self.unreg_mini(y, fbp)
        k = 0
        minimum = False
        while k <= self.total_steps and minimum == False:
            guess_update, fullError = self.update_pic(
                1, self.step_size, y, guess, self.mu_default)
            if ut.l2_norm(guess_update - x_true) >= ut.l2_norm(guess-x_true):
                minimum = True
            else:
                guess = guess_update
        
        data_error, was_output, cut_reco, quality = self.calculate_pic_grad(reconstruction=guess, data_term=y, mu=self.mu_default, ground_truth=x_true)
        check = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]
        if len(check.keys()) == 0:
            step = 0
        else:
            step = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]
        
        ground_truth = torch.tensor(x_true, requires_grad=False, device=self.device)
        noisy_image = torch.tensor(fbp, requires_grad=False, device=self.device)
        quality_noisy = ut.quality(ground_truth, noisy_image)

        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/Data_Loss', data_error.detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/Regulariser_output', was_output.detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/Penalty', fullError.detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/L2_recon_to_ground_truth', quality[0].detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/PSNR_recon_to_ground_truth', quality[1].detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/SSI_recon_to_ground_truth', quality[2].detach().cpu().numpy(), step)

        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/L2_ground_truth_to_noisy', quality_noisy[0].detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/PSNR_ground_truth_to_noisy', quality_noisy[1].detach().cpu().numpy(), step)
        self.Reconstruction_Quality_writer.add_scalar('Reconstruction_Quality/SSI_ground_truth_to_noisy', quality_noisy[2].detach().cpu().numpy(), step)


        self.Reconstruction_Quality_writer.add_image('Reconstruction_Quality/Reconstruction', cut_reco.detach().cpu().numpy()[0], step)
        self.Reconstruction_Quality_writer.add_image('Reconstruction_Quality/Noisy_Image', y[0], step)
        self.Reconstruction_Quality_writer.add_image('Reconstruction_Quality/Ground_truth', x_true[0], step)
        del data_error, was_output, cut_reco, quality
        #torch.cuda.empty_cache()
        
    def log_network_training(self):
        # evaluates and prints the network performance.
        y, x_true, fbp = self.generate_training_data(
            batch_size=self.batch_size, training_data=False)
        guess = np.copy(fbp)
        if self.starting_point == 'Mini':
            guess,fullError = self.unreg_mini(y, fbp=fbp)
        # generate random distribution for rays
        epsilon = np.random.uniform(size=self.batch_size)
        loss, wasserstein_loss, regulariser_was = self.train_step(gen_im=guess, true_im=x_true, random_uint=epsilon, log=True)
        
        check = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]
        if len(check.keys()) == 0:
            step = 0
        else:
            step = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]
        
        self.Network_Optimization_writer.add_scalar('Network_Optimization/Data_Difference', wasserstein_loss.item(), step)
        self.Network_Optimization_writer.add_scalar('Network_Optimization/Lipschitz_Regulariser', regulariser_was.detach().cpu().numpy(), step)
        self.Network_Optimization_writer.add_scalar('Network_Optimization/Overall_Net_Loss', loss.item(), step)


    def log_optimization(self, batch_size=None, steps=None, step_s=None,
                         mu=None, starting_point=None):
        # Logs every step of picture optimization.
        # Can be used to play with the variational formulation once training is complete
        if batch_size is None:
            batch_size = self.batch_size
        if steps is None:
            steps = self.total_steps
        if step_s is None:
            step_s = self.step_size
        if mu is None:
            mu = self.mu_default
        if starting_point is None:
            starting_point = self.starting_point
        y, x_true, fbp = self.generate_training_data(
            batch_size, training_data=False, logOpti=True)
        
        ground_truth = torch.tensor(x_true, requires_grad=False, device=self.device)
        noisy_image = torch.tensor(fbp, requires_grad=False, device=self.device)
        quality_noisy = ut.quality(ground_truth, noisy_image)
        
        print(f"l2: {quality_noisy[0].detach().cpu().numpy()} PSNR: {quality_noisy[1].detach().cpu().numpy()} SSI: {quality_noisy[2].detach().cpu().numpy()}")
        
        guess = np.copy(fbp)
        if starting_point == 'Mini':
            guess, fullError = self.unreg_mini(y, fbp)
        writer = SummaryWriter(
            self.path + '/Logs/Picture_Opt/mu_{}_step_s_{}'.format(mu, step_s))
        for k in range(steps+1):
            #print(f'Optimization with {mu} step: {k}')
            data_error, was_output, cut_reco, quality = self.calculate_pic_grad(reconstruction=guess, data_term=y, mu=mu, ground_truth=x_true)
            
            ground_truth = torch.tensor(x_true, requires_grad=False, device=self.device)
            noisy_image = torch.tensor(y, requires_grad=False, device=self.device)
            quality_noisy = ut.quality(ground_truth, noisy_image)

            
            writer.add_scalar('Picture_Optimization/Data_Loss', data_error.detach().cpu().numpy(), k)
            writer.add_scalar('Picture_Optimization/Regulariser_output', was_output.detach().cpu().numpy(), k)
            writer.add_scalar('Picture_Optimization/Penalty', fullError.detach().cpu().numpy(), k)
            del fullError
            writer.add_scalar('Picture_Optimization/L2_recon_to_ground_truth', quality[0].detach().cpu().numpy(), k)
            writer.add_scalar('Picture_Optimization/PSNR_recon_to_ground_truth', quality[1].detach().cpu().numpy(), k)
            writer.add_scalar('Picture_Optimization/SSI_recon_to_ground_truth', quality[2].detach().cpu().numpy(), k)

            writer.add_scalar('Picture_Optimization/L2_noisy_to_ground_truth', quality_noisy[0].detach().cpu().numpy(), k)
            writer.add_scalar('Picture_Optimization/PSNR_noisy_to_ground_truth', quality_noisy[1].detach().cpu().numpy(), k)
            writer.add_scalar('Picture_Optimization/SSI_noisy_to_ground_truth', quality_noisy[2].detach().cpu().numpy(), k)
            
            writer.add_image('Picture_Optimization/Reconstruction', cut_reco.detach().cpu().numpy()[0], k)
            writer.add_image('Picture_Optimization/Noisy_Image', y[0], k)
            writer.add_image('Picture_Optimization/Ground_truth', x_true[0], k)
            
            guess, fullError = self.update_pic(1, step_s, y, guess, mu)
            del data_error, was_output, cut_reco, quality
            #torch.cuda.empty_cache()
        writer.close()
        self.data_pip.eval_counter = 0
    
    def train_step(self, gen_im, true_im, random_uint, log = False):

        # the network outputs
        gen_im = torch.tensor(gen_im, requires_grad=True, device=self.device)
        true_im = torch.tensor(true_im, requires_grad=False, device=self.device)
        random_uint = torch.tensor(random_uint, device=self.device)
        gen_was = self.network(gen_im)
        data_was = self.network(true_im)

        # Wasserstein loss
        wasserstein_loss = torch.mean(data_was-gen_was)
        
        # intermediate point
        random_uint_exp = torch.unsqueeze(torch.unsqueeze(
            torch.unsqueeze(random_uint, axis=1), axis=1), axis=1)
        
        """
        !!!!! Formual differeing from the paper.#
        original in code : inter = torch.multiply(gen_im, random_uint_exp) + torch.multiply(true_im, 1 - random_uint_exp)
        Paper            : inter = torch.multiply(true_im, random_uint_exp) + torch.multiply(gen_im, 1 - random_uint_exp)
        """

        inter = torch.multiply(gen_im, random_uint_exp) + torch.multiply(true_im, 1 - random_uint_exp)
        
        inter_was = self.network(inter)

        # calculate derivative at intermediate point
        gradient_was = torch.autograd.grad(
            inter_was, inter, grad_outputs=torch.ones_like(inter_was), create_graph=True, retain_graph=True)[0]
        
        # take the L2 norm of that derivative
        norm_gradient = torch.sqrt(torch.sum(torch.square(gradient_was), axis=(1, 2, 3)))

        regulariser_was = torch.mean(torch.square(torch.nn.ReLU()(norm_gradient - 1)))

        # Overall Net Training loss
        loss_was = wasserstein_loss + self.lmb * regulariser_was
        if log:
            del gen_im, true_im, inter_was, inter, gradient_was, norm_gradient, gen_was, data_was
        return loss_was, wasserstein_loss, regulariser_was

    def train(self, steps):
        # the training routine
        for k in range(steps):
            print("step : ", k)
            self.optimizer.zero_grad()
            if k % 100 == 0:
                self.log_network_training()
                self.log_minimum()
            y, x_true, fbp = self.generate_training_data(self.batch_size)
            guess = np.copy(fbp)
            if self.starting_point == 'Mini':
                guess, fullError = self.unreg_mini(y, fbp=fbp)
            # generate random distribution for rays
            epsilon = np.random.uniform(size=self.batch_size)
            # optimize network
            loss, wasserstein_loss, regulariser_was = self.train_step(gen_im=guess, true_im=x_true, random_uint=epsilon)
            loss.backward()
            self.optimizer.step()
            del loss, wasserstein_loss, regulariser_was
            #torch.cuda.empty_cache()
        step = self.optimizer.state[self.optimizer.param_groups[0]["params"][-1]]["step"]
        self.save(step)
        

    def calculate_pic_grad(self, reconstruction, data_term, mu= 0, ground_truth = []):
        """
        reconstruction : x <- A^{dagger}_{deta}y
        data_term : y
        mu : lambda
        """
        # data loss
        if len(ground_truth) == 0:
            reconstruction = torch.tensor(reconstruction, requires_grad=True, device=self.device)
            data_term = torch.tensor(data_term, requires_grad=True, device=self.device)
        else:
            reconstruction = torch.tensor(reconstruction, requires_grad=False, device=self.device)
            data_term = torch.tensor(data_term, requires_grad=False, device=self.device)
        
        mu = torch.tensor(mu, device=self.device)
        
        ray = self.model.tensor_operator(reconstruction) #Ax
        data_mismatch = torch.square(ray - data_term) # Ax - y
        data_error = torch.mean(
            torch.sum(data_mismatch, axis=(1, 2, 3)))

        # the loss functional
        was_output = torch.mean(self.network(reconstruction))

        if len(ground_truth) == 0:
            full_error = mu * was_output + data_error

            # get the batch size - all gradients have to be scaled by the batch size as they are taken over previously
            # averaged quantities already. Makes gradients scaling batch size inveriant
            batch_s = reconstruction.size()[0]
            full_error_batch = full_error * batch_s

            # Optimization for the picture
            pic_grad = torch.autograd.grad(
                full_error_batch, reconstruction)
            pic_grad_cpu = pic_grad[0].detach().cpu().numpy()
            del reconstruction, data_term, data_mismatch, data_error, pic_grad, mu, ray, full_error_batch
            #torch.cuda.empty_cache()
            return pic_grad_cpu, full_error
        
        else:
            cut_reco = torch.clamp(reconstruction, 0.0, 1.0)
            ground_truth = torch.tensor(ground_truth, requires_grad=False, device=self.device)
            quality = ut.quality(ground_truth, reconstruction)
            #quality = torch.mean(torch.sqrt(torch.sum(torch.square(ground_truth - reconstruction),axis=(1, 2, 3))))
            return data_error, was_output, cut_reco, quality
        

    def find_good_lambda(self, sample=64):
        # Method to estimate a good value of the regularisation paramete.
        # This is done via estimation of 2 ||K^t (Kx-y)||_2 where x is the ground truth
        y, x_true, fbp = self.generate_training_data(sample)
        gradient_truth, fullError = self.calculate_pic_grad(reconstruction=x_true, data_term=y, mu=0)
        print('Value of mu around equilibrium: ' + str(np.mean(np.sqrt(np.sum(
              np.square(gradient_truth), axis=(1, 2, 3))))))
        return np.mean(np.sqrt(np.sum(np.square(gradient_truth), axis=(1, 2, 3))))

    def evaluate(self, guess, measurement):
        fbp = np.copy(guess)
        if self.starting_point == 'Mini':
            fbp,fullError = self.unreg_mini(measurement, fbp)
        return self.update_pic(steps=self.total_steps, measurement=measurement, guess=fbp,
                               stepsize=self.step_size, mu=self.mu_default)
