import numpy as np

from . import DataOperatorInterface


class FlippingDataOperator(DataOperatorInterface):
    def __init__(self):
        """
        Flipping transform
        """
        self.direction = None

        self.flip_function = {'left_right': np.fliplr, 'up_down': np.flipud}

    def set_operation_parameters(self, direction):
        """
        Manually set the flipping direction. Allowed options: left_right | up_down
        :param direction: str. left_right | up_down
        """
        self.direction = direction

    def run(self, data, generate_random_parameters=True):
        if generate_random_parameters:
            self.direction = np.random.choice(['left_right', 'up_down'])
        else:
            assert self.direction is not None, "Flipping direction (left_right | up_down) not specified"

        if isinstance(data, np.ndarray):
            return self.flip_function[self.direction](data)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(self.flip_function[self.direction](image))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
