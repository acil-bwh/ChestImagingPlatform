class DataAugmentorInterface(object):
    def generate_augmented_data_points(self, xs, ys, n=1):
        """
        Generate n input/output augmented data points from an input/output data point
        :param xs: list of numpy arrays (inputs).
        :param ys: list of numpy arrays (outputs).
        :param n: number of augmented data points
        :return: Tuple of 2 elements:
            - augmented_xs: List of numpy arrays. Each array will have a size of n x xs[i].shape
            - augmented_xs: List of numpy arrays. Each array will have a size of n x ys[i].shape
        """
        raise NotImplementedError("This method must be implemented in a child class")