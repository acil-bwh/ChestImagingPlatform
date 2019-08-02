import numpy as np
from skimage import transform

from . import DataOperatorInterface


class ShearDataOperator(DataOperatorInterface):
    def __init__(self, min_shear, max_shear, cval=0.0):
        """
        Shear transform.
        Use None for the parameters that are not going to be used
        :param min_shear: int. Min shear angle in degrees
        :param max_shear: int. Max shear angle in degrees
        :param cval: float. Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
        """
        self.min_shear = min_shear
        self.max_shear = max_shear

        self.cval = cval
        self.shear_angle = None

    def set_operation_parameters(self, shear_angle):
        """
        Manually set the parameters needed to apply the operation
        :param shear_angle: int. Shear angle in degrees
        """
        self.shear_angle = np.deg2rad(shear_angle)

    def run(self, data, generate_random_parameters=True):
        """
        Run the operation.
        :param data: Numpy array of float or list of numpy arrays
        :param generate_random_parameters: use the class policy to generate the parameters randomly.
        :return: numpy array (if 'data' is a single numpy array) or list of numpy arrays
        """
        if generate_random_parameters:
            if self.min_shear is not None:
                self.shear_angle = np.deg2rad(np.random.randint(low=self.min_shear, high=self.max_shear))
        else:
            assert self.shear_angle is not None, "Shear angle not specified"

        tr = transform.AffineTransform(translation=None, rotation=None, scale=None, shear=self.shear_angle)

        if isinstance(data, np.ndarray):
            return transform.warp(data, tr, cval=self.cval, preserve_range=True)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(transform.warp(image, tr, cval=self.cval, preserve_range=True))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
