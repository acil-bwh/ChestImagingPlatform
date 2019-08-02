import numpy as np

from ...dcnn.data.data_processing import DataProcessing
from . import DataOperatorInterface


class PerspectiveSkew2DTransformDataOperator(DataOperatorInterface):
    def __init__(self, min_skew, max_skew, skew_type='random', resamplig_filter='bicubic'):
        """
        Apply perspective skewing on images
        :param min_skew: int. Min skew amount
        :param max_skew: int. Max skew amount
        :param skew_type: str. Skew type. Options: random | tilt (will randomly skew either left, right, up, or down.) |
        tilt_top_buttton (skew up or down) | tilt_left_right (skew left or right) |
        corner (will randomly skew one **corner** of the image either along the x-axis or y-axis. Default is random.
        :param resamplig_filter: str. Resampling filter. Options: nearest (use nearest neighbour) |
        bilinear (linear interpolation in a 2x2 environment) | bicubic (cubic spline interpolation in a 4x4 environment)
        """
        self.min_skew = min_skew
        self.max_skew = max_skew

        self.skew_type = skew_type
        self.resamplig_filter = resamplig_filter

        self.skew_amount = None

    def set_operation_parameters(self, skew_amount):
        """
        Manually set the parameters needed to apply the operation
        :param skew_amount: int. The degree to which the image is skewed.
        """
        self.skew_amount = skew_amount

    def run(self, data, generate_random_parameters=True):
        if generate_random_parameters:
            # Generate parameters
            self.skew_amount = np.random.randint(self.min_skew, self.max_skew)
        else:
            assert self.skew_amount is not None, "Skew amount not specified"

        if isinstance(data, np.ndarray):
            return DataProcessing.perspective_skew_2D_transform(data, self.skew_amount, resampling=self.resamplig_filter)
        elif isinstance(data, list):
            result = list()
            for image in data:
                result.append(
                    DataProcessing.perspective_skew_2D_transform(image, self.skew_amount, skew_type=self.skew_type,
                                                                 resampling=self.resamplig_filter))
            return result
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")


