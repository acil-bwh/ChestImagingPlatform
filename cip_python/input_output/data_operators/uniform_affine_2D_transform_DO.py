import math
import numpy as np
from skimage import transform

from . import DataOperatorInterface

class UniformAffineTransform2DDataOperator(DataOperatorInterface):
    def __init__(self, padding, min_translation_h, max_translation_h, min_translation_v, max_translation_v, 
                 min_rotation, max_rotation, min_scale_h, max_scale_h, min_scale_v, max_scale_v, min_shear, max_shear):
        """
        Apply an AffineTranform in 2D using an uniform parameters distribution.
        Use None for the parameters that are not going to be used
        :param padding: float. Value that will be used to fill the padding needed in the transformation
        :param min_translation_h: float. Min translation in horizontal [0-1]
        :param max_translation_h: float
        :param min_translation_v: float. Min translation in vertical [0-1]
        :param max_translation_v: float
        :param min_rotation: float. Rotation in degrees
        :param max_rotation: float
        :param min_scale: float
        :param max_scale: float
        :param min_shear: float. Shear angle in degrees
        :param max_shear: float
        """   
        self.min_translation_h = min_translation_h
        self.max_translation_h = max_translation_h
        self.min_translation_v = min_translation_v
        self.max_translation_v = max_translation_v
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.min_scale_h = min_scale_h
        self.max_scale_h = max_scale_h
        self.min_scale_v = min_scale_v
        self.max_scale_v = max_scale_v
        self.min_shear = min_shear
        self.max_shear = max_shear

        self.padding = padding
        self.translation_h = 0
        self.translation_v = 0
        self.rotation = 0
        self.scale_h = 1
        self.scale_v = 1
        self.shear = 0
        
    def set_operation_parameters(self, rotation, translation_h, translation_v, scale_h, scale_v, shear):
        """
        Manually set the parameters needed to apply the operation
        :param rotation: float. Rotation (in degrees)
        :param translation: float
        :param scale: float
        :param shear: float
        """
        self.rotation = np.deg2rad(rotation)
        self.translation_h = translation_h
        self.translation_v = translation_v
        self.scale_h = scale_h
        self.scale_v = scale_v
        self.shear = np.deg2rad(shear)

    def run(self, data, generate_random_parameters=True):
        """
        Run the operation.
        :param data: Numpy array of float or list of numpy arrays
        :param generate_random_parameters: use the class policy to generate the parameters randomly.
        :return: numpy array (if 'data' is a single numpy array) or list of numpy arrays
        """
        if isinstance(data, np.ndarray):
            image = data
            is_list = False
        elif isinstance(data, list) or isinstance(data, tuple):
            image = data[0]
            is_list = True
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")

        if generate_random_parameters:
            if self.min_rotation is not None:
                self.rotation = np.random.uniform(low=self.min_rotation, high=self.max_rotation) * math.pi / 180.0
            if self.min_translation_h is not None:
                self.translation_h = np.random.uniform(low=self.min_translation_h, high=self.max_translation_h) * \
                                     image.shape[0]
            if self.min_translation_v is not None:
                self.translation_v = np.random.uniform(low=self.min_translation_v, high=self.max_translation_v) * \
                                     image.shape[1]
            if self.min_scale_h is not None:
                self.scale_h = np.random.uniform(low=self.min_scale_h, high=self.max_scale_h) * image.shape[0]
            if self.min_scale_v is not None:
                self.scale_v = np.random.uniform(low=self.min_scale_v, high=self.max_scale_v) * image.shape[1]
            if self.min_shear is not None:
                self.shear = np.random.uniform(low=self.min_shear, high=self.shear) * math.pi / 180.0

        tr = transform.AffineTransform(translation=(self.translation_h, self.translation_v), rotation=self.rotation,
                                       scale=(self.scale_h, self.scale_v), shear=self.shear)

        if is_list:
            result = []
            for image in data:
                result.append(transform.warp(image, tr, cval=self.padding, preserve_range=True))
            return result
        return transform.warp(image, tr, cval=self.padding, preserve_range=True)

