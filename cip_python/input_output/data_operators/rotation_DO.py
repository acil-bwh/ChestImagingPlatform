import numpy as np
from scipy.ndimage.interpolation import rotate

from . import DataOperatorInterface


class RotationDataOperator(DataOperatorInterface):
    def __init__(self, min_rotation_angle, max_rotation_angle, rotation_axes=(1, 0), fill_mode='constant', cval=0.0,
                 order=3, reshape=False):
        """
        Rotate an array in the plane defined by the two axes given by the rotation_axes parameter using spline
        interpolation of the requested order.
        :param min_rotation_angle: int. Max rotation angle in degrees
        :param max_rotation_angle: int. Min rotation angle in degrees
        :param rotation_axis: The two axes that define the plane of rotation. Default is the first two axes.
        :param fill_mode: str. Points outside the boundaries of the input are filled according to the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). Default is ‘constant’.
        :param cval: float. Value used for points outside the boundaries of the input if mode='constant'. Default is 0.0
        :param order: int. The order of the spline interpolation, default is 3. The order has to be in the range 0-5.
        :param reshape: bool. If reshape is True, the output shape is adapted so that the input array is contained completely in the output. Default is False.
        """

        self.max_rotation_angle = max_rotation_angle
        self.min_rotation_angle = min_rotation_angle
        self.rotation_axes = rotation_axes
        self.fill_mode = fill_mode
        self.reshape = reshape
        self.order = order
        self.cval = cval

        self.rotation_angle = None

    def set_operation_parameters(self, rotation_angle):
        """
        Manually set the parameters needed to apply the operation
        :param rotation_angle: int. The rotation angle in degrees
        """
        self.rotation_angle = rotation_angle

    def run(self, data, generate_random_parameters=True):
        if generate_random_parameters:
            self.rotation_angle = np.random.randint(low=self.min_rotation_angle, high=self.max_rotation_angle)
        else:
            assert self.rotation_angle is not None, "Rotation angle not specified"

        if isinstance(data, np.ndarray):
            return rotate(data, self.rotation_angle, axes=self.rotation_axes, mode=self.fill_mode,
                          reshape=self.reshape)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(rotate(image, self.rotation_angle, axes=self.rotation_axes, mode=self.fill_mode,
                                     reshape=self.reshape, cval=self.cval, order=self.order))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
