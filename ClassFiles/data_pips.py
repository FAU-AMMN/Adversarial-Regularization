import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from ClassFiles import util as ut

# Abstract class for data preprocessing. To customize to your own dataset, define subclass with the
# image_size, name and color of your dataset and the corresponding load_data method
class data_pip(ABC):
    image_size = (128,128)
    name = 'default'
    colors = 1

    # Data has to be in path/Training_Data and path/Evaluation_Data
    def __init__(self, path):
        self.data_path = path

    # load data outputs single image in format (image_size, colors).
    # The image should be normalized between (0,1).
    # The training_data flag determines if the image should be taken from training or test set
    @abstractmethod
    def load_data(self, training_data=True):
        pass

# returns 128x128 image from the BSDS dataset.
class BSDS(data_pip):
    name = 'BSDS'
    colors = 3

    def __init__(self, path):
        super(BSDS, self).__init__(path)
        # set up the training data file system
        self.train_list = ut.find('*.jpg', self.data_path+'Training_Data')
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
    def load_data(self, training_data= True):
        pic = self.single_image(training_data=training_data)
        size = pic.shape
        ul, lr = self.edgepoint(size[0], size[1])
        image = pic[ul[0]:lr[0], ul[1]:lr[1],:]
        return image
