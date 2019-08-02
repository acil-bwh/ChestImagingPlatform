import numpy as np
from scipy.ndimage import shift

from . import DataOperatorInterface


class ShiftingDataOperator(DataOperatorInterface):
    def __init__(self, min_shift_x, max_shift_x, min_shift_y, max_shift_y, min_shift_z, max_shift_z,
                 order=3, fill_mode='nearest', cval=0.0):
        """
        Shift an array. The array is shifted using spline interpolation of the requested order.
        Points outside the boundaries of the input are filled according to the given fill_mode
        :param min_shift_x: int. The min shift along X
        :param max_shift_x: int. The min shift along Y
        :param min_shift_y: int. The min shift along Z
        :param max_shift_y: int. The max shift along X
        :param min_shift_z: int. The max shift along Y
        :param max_shift_z: int. The max shift along Z
        :param order: The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        :param fill_mode: The mode parameter determines how the input array is extended beyond its boundaries.
        :param cval: Value to fill past edges of input if mode is ‘constant’. Default is 0.0
        """
        self.min_shift_x = min_shift_x
        self.max_shift_x = max_shift_x
        self.min_shift_y = min_shift_y
        self.max_shift_y = max_shift_y
        self.min_shift_z = min_shift_z
        self.max_shift_z = max_shift_z
        self.fill_mode = fill_mode
        self.order = order
        self.cval=cval

        self.shift_x = None
        self.shift_y = None
        self.shift_z = None

    def set_operation_parameters(self, shift_x, shift_y, shift_z):
        """
        Manually set the parameters needed to apply the operation
        :param shift_x: int. Shift along X
        :param shift_y: int. Shift along Y
        :param shift_z: int. Shift along Z
        :return:
        """
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.shift_z = shift_z

    def run(self, data, generate_random_parameters=True):
        if generate_random_parameters:
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
                result.append(shift(image, (self.shift_x, self.shift_y, self.shift_z), order=self.order,
                                    mode=self.fill_mode, cval=self.cval))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
