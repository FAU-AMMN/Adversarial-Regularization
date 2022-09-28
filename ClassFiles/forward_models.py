import numpy as np
from abc import ABC, abstractmethod
import odl

class ForwardModel(ABC):
    # Defining the forward operators used. For customization, create a subclass of forward_model, implementing
    # the abstract classes.
    name = 'abstract'

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def get_image_size(self):
        # Returns the image size in the format (width, height)
        pass

    @abstractmethod
    def get_measurement_size(self):
        # Returns the measurement size in the format (width, height)
        pass

    # All inputs to the evaluation methods have the format [width, height, channels]
    @abstractmethod
    def forward_operator(self, image):
        # The forward operator
        pass

    @abstractmethod
    def forward_operator_adjoint(self, measurement):
        # Needed for implementation of  RED only.
        # Returns the adjoint of the forward operator of measurements
        pass

    @abstractmethod
    def inverse(self, measurement):
        # An approximate (possibly regularized) inverse of the forward operator.
        # Used as starting point and for training
        pass

    @abstractmethod
    def get_odl_operator(self):
        # The forward operator as odl operator. Needed for total variation only.
        pass

    # Input in the form [batch, width, height, channels]
    @abstractmethod
    def tensor_operator(self, tensor):
        # The forward operator as tensorflow layer. Needed for evaluation during training
        pass

class Denoising(ForwardModel):
    name = 'Denoising'

    def __init__(self, size):
        super(Denoising, self).__init__(size)
        self.size = size
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.size[0], self.size[1]],
                                       dtype='float32')
        self.operator = odl.IdentityOperator(self.space)

    def get_image_size(self):
        return self.size

    def get_measurement_size(self):
        return self.size

    def forward_operator(self, image):
        return image

    def forward_operator_adjoint(self, measurement):
        return measurement

    def inverse(self, measurement):
        return measurement

    def tensor_operator(self, tensor):
        return tensor

    def get_odl_operator(self):
        return self.operator
