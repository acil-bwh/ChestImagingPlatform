import numpy as np
from scipy.ndimage.interpolation import rotate

from . import DataOperatorInterface


class RotationDataOperator(DataOperatorInterface):
    def __init__(self, min_rotation_angle, max_rotation_angle, rotation_axis=(1, 0), fill_mode='nearest', reshape=False):
        self.max_rotation_angle = max_rotation_angle
        self.min_rotation_angle = min_rotation_angle
        self.rotation_axis = rotation_axis
        self.fill_mode = fill_mode
        self.reshape = reshape

        self.rotation_angle = None

    def set_operation_parameters(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def run(self, data, generate_parameters=True):
        if generate_parameters:
            self.rotation_angle = np.random.randint(low=self.min_rotation_angle, high=self.max_rotation_angle)
        else:
            assert self.rotation_angle is not None, "Rotation angle not specified"

        if isinstance(data, np.ndarray):
            return rotate(data, self.rotation_angle, axes=self.rotation_axis, mode=self.fill_mode,
                          reshape=self.reshape)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(rotate(image, self.rotation_angle, axes=self.rotation_axis, mode=self.fill_mode,
                                     reshape=self.reshape))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
