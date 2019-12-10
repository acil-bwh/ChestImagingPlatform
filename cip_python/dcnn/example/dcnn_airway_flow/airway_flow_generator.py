import numpy as np
import threading

from cip_python.dcnn.data import DataProcessing

class AirwayFlowDataGenerator:
    def __init__(self, x_dataset, y_dataset, x_mean, x_std, y_mean, y_std, batch_size=64,
                 num_augmented_data=0, shuffle=False):

        self.x_dataset = x_dataset
        self.y_dataset = y_dataset
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.x_std = x_std
        self.y_std = y_std

        self.batch_size = batch_size

        self.data_shape = self.x_dataset.shape[0]

        self.num_augmented_data = num_augmented_data

        self.lock = threading.Lock()

        self.current_index = 0
        self.num_epochs = 0

        self.data_processing = DataProcessing()

        self.shuffle = shuffle
        self._init_sorting_()

    @property
    def data_after_augmentation(self):
        """
        Total number of images including the augmented data
        """
        return self.data_shape + self.data_shape * self.num_augmented_data

    @staticmethod
    def normalize_data(array, mean, std):
        return (array - mean) / std

    def _init_sorting_(self):
        """
        Set to 0 the index of the image and increment num of epochs
        Returns:
        """
        if self.shuffle:
            pp = np.random.permutation(self.data_shape)
            self.x_dataset = self.x_dataset[pp]
            self.y_dataset = self.y_dataset[pp]

        self.current_index = 0
        self.num_epochs += 1

    def data_generator(self):
        if not self.batch_size:
            raise Exception("Please set batch_size property in the dataset")
        while True:
            x, y = self.next_batch(self.batch_size)
            yield x, y

    def next_batch(self, batch_size):
        x_data = np.zeros((batch_size, ) + self.x_dataset.shape[1:], np.float32)
        y_data = np.zeros((batch_size, ) + self.y_dataset.shape[1:], np.float32)

        batch_position = 0
        while batch_position < batch_size:
            with self.lock:  # Lock for multi-threading
                if self.current_index >= self.data_shape:
                    # New epoch starts
                    self._init_sorting_()

            x_data[batch_position:batch_position+1] = self.x_dataset[self.current_index].astype(np.float32)
            y_data[batch_position:batch_position+1] = self.y_dataset[self.current_index].astype(np.float32)

            batch_position += 1

            num_augmented = min(self.num_augmented_data, batch_size - batch_position)
            for ii in range(1, num_augmented+1):
                mass_flow_noise = np.random.uniform(-0.000001, 0.000001)
                augmented_data = self.x_dataset[self.current_index].astype(np.float32)
                augmented_data[-1] += mass_flow_noise

                x_data[batch_position:batch_position + ii] = augmented_data
                y_data[batch_position:batch_position + ii] = self.y_dataset[self.current_index].astype(np.float32)
                batch_position += ii

            with self.lock:
                self.current_index += 1

        x_data_norm = self.data_processing.standardization(x_data, mean_value=self.x_mean, std_value=self.x_std)
        y_data_norm = self.data_processing.standardization(y_data, mean_value=self.x_mean, std_value=self.x_std)
        return x_data_norm, y_data_norm
