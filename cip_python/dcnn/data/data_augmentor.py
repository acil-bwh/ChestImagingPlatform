import numpy as np

class DataAugmentor(object):
    def __init__(self, data_operators):
        """
        Constructor
        :param data_operators: List of objects of the class DataOperator that will compound the pipeline.
                               In order to generate the data, all the operator will be run sequentially as a pipeline
        """
        self.data_operators = data_operators


    def generate_augmented_data_points(self, xs, ys, n=1, augment_ys=True):
        """
        Generate n input/output augmented data points from an input/output data point.
        All the operator will be executed sequentially as a pipeline
        :param xs: list of numpy arrays (inputs).
        :param ys: list of numpy arrays (outputs).
        :param n: int. number of augmented data points
        :param augment_ys: bool. Augment the ys in the same way the xs are augmented. If False, just
                           create a copy of the original ys for each augmented data point
        :return: Tuple of 2 elements:
            - augmented_xs: List of numpy arrays. Each array will have a size of n x xs[i].shape
            - augmented_xs: List of numpy arrays. Each array will have a size of n x ys[i].shape
        """
        # Reserve space for all the arrays
        augmented_xs = []
        augmented_ys = []
        for input_ in xs:
            augmented_xs.append(np.zeros(((n,) + input_.shape), dtype=np.float32))

        for output in ys:
            augmented_ys.append(np.zeros(((n,) + output.shape), dtype=np.float32))

        for data_point_ix in range(n):
            data = xs
            if augment_ys:
                data.extend(ys)
            for data_operator in self.data_operators:
                data = data_operator.run(data)
            for n in range(len(xs)):
                augmented_xs[n][data_point_ix] = data[n]
            for n in range(len(ys)):
                augmented_ys[n][data_point_ix] = data[len(xs) + n] if augment_ys else np.copy(ys[n])
        return augmented_xs, augmented_ys
