import numpy as np

from ...dcnn.data.data_processing import DataProcessing
from . import DataOperatorInterface


class Elastic2DTransformDataOperator(DataOperatorInterface):
    def __init__(self, min_grid_width, max_grid_width, min_grid_height, max_grid_height,
                 min_magnitude, max_magnitude, resamplig_filter='bicubic'):
        """
        Apply elastic transformation to 2D images.
        :param min_grid_width: int. Min grid width
        :param max_grid_width:  int. Max grid width
        :param min_grid_height: int. Min grid height
        :param max_grid_height: int. Max grid height
        :param min_magnitude: int. Min magnitude
        :param max_magnitude: int. Max magnitude
        :param resamplig_filter: str. Resampling filter. Options: nearest (use nearest neighbour) |
        bilinear (linear interpolation in a 2x2 environment) | bicubic (cubic spline interpolation in a 4x4 environment)
        """
        self.min_grid_width = min_grid_width
        self.max_grid_width = max_grid_width
        self.min_grid_height = min_grid_height
        self.max_grid_height = max_grid_height
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude

        self.resamplig_filter = resamplig_filter

        self.grid_width = self.grid_height = self.magnitude = None

    def set_operation_parameters(self, grid_width, grid_height, magnitude):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.magnitude = magnitude

    def run(self, data, generate_random_parameters=True):
        if generate_random_parameters:
            # Generate parameters
            self.grid_width = np.random.randint(self.min_grid_width, self.max_grid_width)
            self.grid_height = np.random.randint(self.min_grid_height, self.max_grid_height)
            self.magnitude = np.random.randint(self.magnitude, self.magnitude)
        else:
            assert self.grid_width is not None and self.grid_height is not None and self.magnitude is not None, "Grid width, grid height and magnitude not specified"

        if isinstance(data, np.ndarray):
            return DataProcessing.elastic_deformation_2D(data, self.grid_width, self.grid_height, self.magnitude,
                                                         self.resamplig_filter)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(
                    DataProcessing.elastic_deformation_2D(image, self.grid_width, self.grid_height, self.magnitude,
                                                          self.resamplig_filter))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")
