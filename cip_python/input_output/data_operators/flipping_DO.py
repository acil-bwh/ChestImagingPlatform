import numpy as np

from . import DataOperatorInterface


class FlippingDataOperator(DataOperatorInterface):
    def __init__(self):
        self.direction = None

        self.flip_function = {'left_right': np.flipud, 'up_down': np.fliplr}

    def set_operation_parameters(self, direction):
        self.direction = direction

    def run(self, data, generate_parameters=True):
        if generate_parameters:
            self.direction = np.random.choice(['left_right', 'up_down'])
        else:
            assert self.direction is not None, "Flipping direction (left_right | up_down) not specified"

        if isinstance(data, np.ndarray):
            return self.flip_function[self.direction](data)
        elif isinstance(data, list):
            result = list()
            for image in data:
                self.flip_function[self.direction](image)
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
