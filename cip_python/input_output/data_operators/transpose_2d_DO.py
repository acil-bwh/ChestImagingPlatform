import numpy as np

from . import DataOperatorInterface


class Transpose2DDataOperator(DataOperatorInterface):
    def __init__(self):
        """
        Transpose transform
        """
        pass

    def set_operation_parameters(self):
        """
        Manually set the flipping direction. Allowed options: left_right | up_down
        :param direction: str. left_right | up_down
        """
        pass

    def run(self, data, generate_random_parameters=False):
        if isinstance(data, np.ndarray):
            return np.transpose(data)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(np.transpose(image))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
