import random
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from ClassFiles import util as ut
import pydicom as dc
from skimage.transform import resize
import odl

# Abstract class for data preprocessing. To customize to your own dataset, define subclass with the
# image_size, name and color of your dataset and the corresponding load_data method
class data_pip(ABC):
    image_size = (128,128)
    name = 'default'
    colors = 1

    # Data has to be in path/Training_Data and path/Evaluation_Data
    def __init__(self, path, image_size = None):
        self.data_path = path
        if not (image_size is None):
            self.image_size = image_size
    # load data outputs single image in format (image_size, colors).
    # The image should be normalized between (0,1).
    # The training_data flag determines if the image should be taken from training or test set
    @abstractmethod
    def load_data(self, training_data=True):
        pass


class BSDS(data_pip):
    name = 'BSDS'
    colors = 3

    def __init__(self, path, image_size = None):
        super(BSDS, self).__init__(path, image_size)
        # set up the training data file system
        self.train_list = ut.find('*.jpg', self.data_path+'Training_Data') #imgDim=481*321
        self.train_amount = len(self.train_list)
        print('Training Pictures found: ' + str(self.train_amount))
        self.eval_list = ut.find('*.jpg', self.data_path+'Evaluation_Data')
        self.eval_amount = len(self.eval_list)
        print('Evaluation Pictures found: ' + str(self.eval_amount))
        self.eval_counter = 0
    # method to draw raw picture samples
    def single_image(self, training_data=True):
        if training_data:
            self.eval_counter = 0
            rand = random.randint(0, self.train_amount - 1)
            pic = plt.imread(self.train_list[rand])
        else:
            #rand = random.randint(0, self.eval_amount - 1)
            #pic = scipy.ndimage.imread(self.eval_list[rand])
            pic = plt.imread(self.eval_list[self.eval_counter])
            #print(f'image {self.eval_counter},  {self.eval_list[self.eval_counter]}')
            self.eval_counter +=1
        return pic/255.0

    # Draw random edgepoint
    def edgepoint(self, x_size, y_size):
        x_vary = x_size - self.image_size[0]
        #x_coor = random.randint(0, x_vary)
        x_coor = x_vary
        y_vary = y_size - self.image_size[1]
        #y_coor = random.randint(0, y_vary)
        y_coor = y_vary
        upper_left = [x_coor, y_coor]
        lower_right = [x_coor + self.image_size[0], y_coor + self.image_size[1]]
        return upper_left, lower_right

    # methode to cut a image_size area out of the training images
    def load_data(self, training_data= True, logOpti = False):
        pic = self.single_image(training_data=training_data)
        pic = ut.normalize_image(pic)
        if logOpti == False:
            pic = resize(pic, self.image_size)
        pic = ut.scale_to_unit_intervall(pic)
        #size = pic.shape
        #ul, lr = self.edgepoint(size[0], size[1])
        #image = pic[ul[0]:lr[0], ul[1]:lr[1],:]
        return pic


class ellipses(data_pip):
    name = 'ellipses'
    colors = 1

    def __init__(self, path, num_imgs = 1600, image_size=None, n_ellipse=50):
        super(ellipses, self).__init__(path, image_size=image_size)
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.image_size[0], self.image_size[1]],
                                  dtype='float32')
        
        # fix data set during initialization
        self.num_imgs = num_imgs
        self.poiss = np.random.poisson(n_ellipse, size = num_imgs) 
        rands = []
        exps = []
        for n in range(num_imgs):
            rands.append(np.random.rand(self.poiss[n], 4))
            exps.append(np.concatenate((np.random.exponential(scale=0.4, size = (self.poiss[n], 1)), np.random.exponential(size=(self.poiss[n],2))),axis=1))
        self.rands = rands
        self.exps = exps


    # generates one random ellipse (based on already drawn random variables)
    def random_ellipse(self, ph_idx, ell_idx, interior=False):
        if interior:
            x_0 = self.rands[ph_idx][ell_idx, 0] - 0.5
            y_0 = self.rands[ph_idx][ell_idx, 1] - 0.5
        else:
            x_0 = 2 * self.rands[ph_idx][ell_idx, 0] - 1.0
            y_0 = 2 * self.rands[ph_idx][ell_idx, 1] - 1.0

        return ((self.rands[ph_idx][ell_idx, 2] - 0.5) * self.exps[ph_idx][ell_idx,0],
                self.exps[ph_idx][ell_idx,1] * 0.2, self.exps[ph_idx][ell_idx,2] * 0.2,
                x_0, y_0,
                self.rands[ph_idx][ell_idx, 3] * 2 * np.pi)

    # generates odl space object with ellipses
    def random_phantom(self, spc, ph_idx, interior=False):
        n = self.poiss[ph_idx]
        ellipses = [self.random_ellipse(ph_idx=ph_idx, ell_idx=j, interior=interior) for j in range(n)]
        return odl.phantom.ellipsoid_phantom(spc, ellipses)

    def load_data(self, training_data= True, logOpti = False):
        if training_data:
            ph_idx = random.randint(0, self.num_imgs-1)
        else:
            ph_idx = 0
        pic = self.random_phantom(spc= self.space, ph_idx=ph_idx)
        output = np.zeros((self.image_size[0], self.image_size[1], 1))
        output[..., 0] = ut.scale_to_unit_intervall(pic)
        return output
