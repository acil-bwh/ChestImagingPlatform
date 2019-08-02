import numpy as np

from . import DataOperatorInterface

class GaussianNoiseDataOperator(DataOperatorInterface):
    def __init__(self, min_noise_mean, max_noise_mean, min_noise_std, max_noise_std):
        self.max_noise_std = max_noise_std
        self.min_noise_std = min_noise_std
        self.max_noise_mean = max_noise_mean
        self.min_noise_mean = min_noise_mean

        self.noise_mean = None
        self.noise_std = None

    def set_operation_parameters(self, noise_mean, noise_std):
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def run(self, data, generate_random_parameters=True):
        if generate_random_parameters:
            self.noise_mean = np.random.uniform(low=self.min_noise_mean, high=self.max_noise_mean)
            self.noise_std = np.random.uniform(low=self.min_noise_std, high=self.max_noise_std)
        else:
            assert self.noise_mean is not None and self.noise_std is not None, "Mean and std not specified"

        if isinstance(data, np.ndarray):
            image = data
            is_list = False
        elif isinstance(data, list):
            image = data[0]
            is_list = True
        else:
            raise AttributeError("Wrong type for 'data'. It should be an image (numpy array) or a list of images")

        noise = np.random.normal(self.noise_mean, self.noise_std, image.shape)

        if is_list:
            # Return a new list
            result = []
            for image in data:
                result.append(image + noise)
            return result
        else:
            return data + noise







