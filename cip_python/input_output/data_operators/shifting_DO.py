import numpy as np
from scipy.ndimage import shift

from . import DataOperatorInterface


class ShiftingDataOperator(DataOperatorInterface):
    def __init__(self, min_shift_x, max_shift_x, min_shift_y, max_shift_y, min_shift_z, max_shift_z,
                 fill_mode='nearest'):
        self.min_shift_x = min_shift_x
        self.max_shift_x = max_shift_x
        self.min_shift_y = min_shift_y
        self.max_shift_y = max_shift_y
        self.min_shift_z = min_shift_z
        self.max_shift_z = max_shift_z
        self.fill_mode = fill_mode

        self.shift_x = None
        self.shift_y = None
        self.shift_z = None

    def set_operation_parameters(self, shift_x, shift_y, shift_z):
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z

    def run(self, data, generate_parameters=True):
        if generate_parameters:
            self.shift_x = np.random.randint(low=self.min_shift_x, high=self.max_shift_x)
            self.shift_y = np.random.randint(low=self.min_shift_y, high=self.max_shift_y)
            self.shift_x = np.random.randint(low=self.min_shift_z, high=self.max_shift_z)
        else:
            assert self.shift_x is not None and self.shift_y is not None and self.shift_z is not None, "Shift along the three dimensions not specified"

        if isinstance(data, np.ndarray):
            return shift(data, (self.shift_x, self.shift_y, self.shift_z), mode=self.fill_mode)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(shift(image, (self.shift_x, self.shift_y, self.shift_z), mode=self.fill_mode))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
