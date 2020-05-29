from __future__ import division
import warnings
import time
import datetime
import h5py
import threading
import numpy as np
import random
import math
import pandas as pd

from sklearn.model_selection import train_test_split

class H5Manager(object):
    # Data points status
    UNDEFINED = 0
    TRAIN = 1
    VALIDATION = 2
    TEST = 3

    def __init__(self, h5_file_path, open_mode='r',
                 load_all_data=False,
                 batch_size=None,
                 shuffle_training=True,
                 train_ixs=None, validation_ixs=None, test_ixs=None,
                 xs_dataset_names=('images',), ys_dataset_names=('labels',),
                 use_pregenerated_augmented_train_data=False,
                 use_pregenerated_augmented_val_data=False,
                 pregenerated_augmented_xs_dataset_names=('images_augmented',),
                 pregenerated_augmented_ys_dataset_names=('labels_augmented',),
                 num_augmented_train_data_points_per_original_data_point=0,
                 ):
        """
        Constructor
        :param h5_file_path: str. Path to the h5 file
        :param open_mode: str. Open mode for the h5 file (defaut: 'r'=read only)
        :param load_all_data: bool. If True all h5 data will be charged into memory before being used.
        :param network: instance of Network class that will receive the dataset data
        :param batch_size: int. Batch size used in keras generators
        :param shuffle_training: bool. Shuffle the training indexes
        :param train_ixs: numpy array of int. Training indexes. They can be initialized later
        :param validation_ixs: numpy array of int. Validation indexes. They can be initialized later
        :param test_ixs: numpy array of int. Test indexes. They can be initialized later
        :param num_augmented_train_data_points_per_original_data_point: int. Number of data points that will be read for
                                                                        each original data point
        :param xs_dataset_names: tuple/list of str. Names for the datasets that will be used as input data
        :param ys_dataset_names: tuple/list of str. Names for the datasets that will be used as output data
        :param use_pregenerated_augmented_train_data: bool. Use a dataset that already contains pre-augmented data
        :param use_pregenerated_augmented_val_data: bool. Use a dataset that already contains pre-augmented data for validation
        :param pregenerated_augmented_xs_dataset_names: tuple/list of str. Names for the datasets that will be used as augmented input data
        :param pregenerated_augmented_ys_dataset_names: tuple/list of str. Names for the datasets that will be used as augmented output data
        """
        self.h5_file_path = h5_file_path

        self.xs_dataset_names = xs_dataset_names
        self.ys_dataset_names = ys_dataset_names

        self.xs_ds_all_data = None
        self.ys_ds_all_data = None
        self.load_all_data = load_all_data
        try:
            self.h5 = h5py.File(h5_file_path, open_mode)
            if load_all_data:
                self.load_all_data_first()
        except Exception as ex:
            raise Exception("H5 File {} could not be opened in '{}' mode: {}".format(h5_file_path, open_mode, ex))

        self._xs_sizes_ = self._ys_sizes_ = None

        self.shuffle_training = shuffle_training
        self.batch_size = batch_size

        # Data augmentation
        self.num_augmented_train_data_points_per_data_point = num_augmented_train_data_points_per_original_data_point
        if self.num_augmented_train_data_points_per_data_point is None:
            self.num_augmented_train_data_points_per_data_point = 0

        # Pregenerated data augmentation
        self.use_pregenerated_augmented_train_data = use_pregenerated_augmented_train_data
        self.use_pregenerated_augmented_val_data = use_pregenerated_augmented_val_data
        self._pregenerated_augmented_ixs_ = None      # 2-Dimensional Array with the indexes of the images generated with data augmentation
        self._pregenerated_augmented_ixs_pos_ = None  # 1-dimensional array with current positions of the list of indexes for the images generated with data augmentation
        self._pregenerated_augmented_ixs_val_ = None  # 2-Dimensional Array with the indexes of the images generated with data augmentation (for validation)
        self._pregenerated_augmented_ixs_val_pos_ = None  # 1-dimensional array with current positions of the list of indexes for the images generated with data augmentation (for validation)
        self.pregenerated_augmented_xs_ds_names = pregenerated_augmented_xs_dataset_names
        self.pregenerated_augmented_ys_ds_names = pregenerated_augmented_ys_dataset_names
        self.pregenerated_num_augmented_data_points_per_data_point = None

        # Indexes
        self.train_ixs = train_ixs
        self._train_ix_pos_ = 0
        self.validation_ixs = validation_ixs
        self._validation_ix_pos_ = 0
        self.test_ixs = test_ixs
        self._test_ix_pos_ = 0

        self._num_epoch_ = 0
        self.epoch_begin = None     # Time where the current epoch began

        self.lock = threading.Lock()

    def load_all_data_first(self):
        self.xs_ds_all_data = dict()
        self.ys_ds_all_data = dict()
        for n in self.xs_dataset_names:
            self.xs_ds_all_data[n] = self.h5[n][:]
        for m in self.ys_dataset_names:
            self.ys_ds_all_data[m] = self.h5[m][:]

    def generate_train_validation_ixs_random(self, num_data_points_used=None, validation_proportion=0.1):
        """
        Generate a random split of the data point for training/validation and save the indexes.
        The results will be stored in self.train_ixs and self.validation_ixs
        :param num_data_points_used: int. Number of data points that will be used (the first elements in the dataset).
                                      If None, all the data points in the dataset will be used
        :param validation_proportion: float 0-1. Proportion of cases reserved for training (the rest will be used for validation)
        """
        if num_data_points_used is None:
            num_data_points_used = self.num_original_data_points
        ixs = np.arange(num_data_points_used)
        self.train_ixs, self.validation_ixs = train_test_split(ixs, test_size=validation_proportion)

    def generate_train_validation_ixs_random_using_key(self, ds_key_name, num_data_points_used=None,
                                                       validation_proportion=0.1):
        """
        Generate a random split of train/validation data using the dataset 'ds_key_name' as a reference.
        For instance, if ds_key_name='cid' and validation_proportion=0.1, the datapoints that belong to
        10% of the cases will be used for validation
        :param ds_key_name: str. Name of the dataset that contains the keys (one entry for each data point)
        :param validation_proportion: float [0-1]. Proportion of the key (cid, sid, etc.) reserverd for validation
        :param num_data_points_used: int. Max index in the dataset that will be searched (excluded).
                               If None, the whole dataset will be used
        """
        n = self.h5[ds_key_name].shape[0] if num_data_points_used is None else num_data_points_used
        all_keys = self.h5[ds_key_name][:n]
        unique_keys = np.unique(all_keys)
        np.random.shuffle(unique_keys)
        num_val_cases = int(len(unique_keys) * validation_proportion)
        val_cases = unique_keys[:num_val_cases]
        # Use dataframes to boost performance
        df = pd.DataFrame(data=all_keys, columns=['id'])
        val_cases_df = pd.DataFrame(data=val_cases, columns=['id']).set_index('id')
        dfj = df.merge(val_cases_df, how='left', left_on='id', right_index=True, indicator=True)
        self.train_ixs = dfj[dfj['_merge'] != 'both'].index.values
        self.validation_ixs = dfj[dfj['_merge'] == 'both'].index.values

    def generate_ixs_from_dataset(self, dataset_name):
        """
        Read the use indexes (train, validation, test) from a dataset
        :param dataset_name: str. Name of the dataset that contain the indexes
        """
        a = self.h5[dataset_name][:]
        self.train_ixs = np.argwhere(a == self.TRAIN)[:, 0]
        self.validation_ixs = np.argwhere(a == self.VALIDATION)[:, 0]
        self.test_ixs = np.argwhere(a == self.TEST)[:, 0]

    @property
    def num_train_points(self):
        return len(self.train_ixs)

    @property
    def num_validation_points(self):
        return len(self.validation_ixs)

    @property
    def num_test_points(self):
        return len(self.test_ixs)

    @property
    def num_original_data_points(self):
        """
        Number of original data points in the dataset.
        Note that we should have at least one input dataset initialized
        :return: int
        """
        assert len(self.xs_dataset_names) > 0, "The dataset names have not been initialized, so there is no way I can give you this info! " \
                                          "Please set a value for the 'xs_dataset_names' property so I can know where I can find the data"
        return self.h5[self.xs_dataset_names[0]].shape[0]


    @property
    def num_data_points_pregenerated_per_original_data_point(self):
        """
        Number of data points that have been pregenerated in a previous data augmentation for each original data point
        :return: int
        """
        if not self.use_pregenerated_augmented_train_data:
            return 0
        assert len(self.pregenerated_augmented_xs_ds_names) > 0, \
            "The pregenerated dataset names have not been initialized, so there is no way I can give you this info! " \
            "Please set a value for the 'pregenerated_augmented_xs_ds_names' property so I can know where I can find the data"
        return self.h5[self.pregenerated_augmented_xs_ds_names[0]].shape[1]

    @property
    def num_total_train_data_points_per_epoch(self):
        """
        Num data points used for training including the data augmentation (pregenerated or on the fly) for a complete epoch
        """
        return self.num_train_points + (self.num_train_points * self.num_augmented_train_data_points_per_data_point)

    @property
    def current_epoch(self):
        """
        Current number of epoch
        :return: int
        """
        return self._num_epoch_

    @property
    def xs_sizes(self):
        """
        List of sizes for each one of the inputs (xs)
        :return: list
        """
        if self._xs_sizes_ is None:
            self._xs_sizes_ = list(map(lambda name: self.h5[name].shape[1:], self.xs_dataset_names))
        return self._xs_sizes_

    @property
    def ys_sizes(self):
        """
        List of sizes for each one of the inputs (xs)
        :return: list
        """
        if self._ys_sizes_ is None:
            self._ys_sizes_ = list(map(lambda name: self.h5[name].shape[1:], self.ys_dataset_names))
        return self._ys_sizes_


    def _training_sanity_checks_(self):
        """
        Make sure that all the required datasets (for training) are found in the h5 file
        """
        for ds_name in self.xs_dataset_names:
            assert ds_name in self.h5, "'{}' dataset not found".format(ds_name)
        for ds_name in self.ys_dataset_names:
            assert ds_name in self.h5, "'{}' dataset not found".format(ds_name)
        if self.use_pregenerated_augmented_train_data:
            for ds_name in self.pregenerated_augmented_xs_ds_names:
                assert ds_name in self.h5, "'{}' dataset not found".format(ds_name)
            for ds_name in self.pregenerated_augmented_ys_ds_names:
                assert ds_name in self.h5, "'{}' dataset not found".format(ds_name)
            assert len(self.xs_dataset_names) == len(self.pregenerated_augmented_xs_ds_names), \
                "The original input datasets number and the augmented input datasets number should match"
            assert len(self.ys_dataset_names) == len(self.pregenerated_augmented_ys_ds_names), \
                "The original labels datasets number and the augmented labels datasets number match"

    def get_all_train_data(self):
        """Get all the training data points
        :return: Tuple  of tuples with xs, ys
        """
        assert self.train_ixs is not None, "Train indexes not initialized"
        return self.get_next_batch(self.num_train_points, self.TRAIN)

    def get_all_validation_data(self):
        """Get all the validation data points
        :return: Tuple  of tuples with xs, ys
        """
        assert self.validation_ixs is not None, "Validation indexes not initialized"
        return self.get_next_batch(self.num_validation_points, self.VALIDATION)

    def get_all_test_data(self):
        """Get all the test data points
        :return: Tuple  of tuples with xs, ys
        """
        assert self.test_ixs is not None, "Test indexes not initialized"
        return self.get_next_batch(self.num_test_points, self.TEST)

    def _new_epoch_(self):
        """
        Beginning of an epoch. Initialize and shuffle indexes if needed
        """
        if self.current_epoch == 0:
            # Begging of the training
            self._training_sanity_checks_()
            assert (self.train_ixs is not None
                    and isinstance(self.train_ixs, np.ndarray)
                    and self.train_ixs.dtype == np.int), \
                "You must set the training indexes (numpy of ints) using one of these methods: \n" \
                "- 'generate_random_train_validation_partition' method\n" \
                "- 'read_datapoint_use_ixs_from_dataset' method\n" \
                "- Manual asignation of 'train_ixs' variable"

        with self.lock:
            print("\n\n######## New Dataset epoch ({}) ###########".format(self._num_epoch_ + 1))
            t2 = time.time()
            if self.epoch_begin is not None:
                total_seconds = t2 - self.epoch_begin
                print("Previous epoch time: {}".format(datetime.timedelta(seconds=total_seconds)))

            self.epoch_begin = t2
            if self.shuffle_training:
                random.shuffle(self.train_ixs)

            if self.use_pregenerated_augmented_train_data:
                if self._pregenerated_augmented_ixs_ is None:
                    # We need to create a list for each index in the training dataset.
                    # In order to loop over all the images in the dataset, we need main index
                    # that contains 'num_total_augmented_images' for each main image.
                    # We assume we have the same number of augmented data points for each used data dataset
                    if self.pregenerated_num_augmented_data_points_per_data_point is None:
                        # Obtain from dataset directly
                        self.pregenerated_num_augmented_data_points_per_data_point = self.h5[self.pregenerated_augmented_xs_ds_names[0]].shape[1]
                    self._pregenerated_augmented_ixs_ = np.tile(np.arange(self.pregenerated_num_augmented_data_points_per_data_point),
                                                                self.num_train_points).reshape(self.num_train_points, -1)
                if self.shuffle_training:
                    for i in range(self.num_train_points):
                        np.random.shuffle(self._pregenerated_augmented_ixs_[i])

            self._train_ix_pos_ = 0
            self._pregenerated_augmented_ixs_pos_ = [0] * self.num_train_points
            self._num_epoch_ += 1

    def read_data_point(self, main_ix):
        """
        Read xs, ys for a particular index
        :param main_ix: index in the h5 file (main index)
        :return: tuple of xs,ys in the dataset format
        """
        num_xs = len(self.xs_sizes)
        num_ys = len(self.ys_sizes)
        xs = [None] * num_xs
        ys = [None] * num_ys

        if not self.load_all_data:
            xs_ds = tuple(map(lambda n: self.h5[n], self.xs_dataset_names))
            ys_ds = tuple(map(lambda n: self.h5[n], self.ys_dataset_names))
        else:
            xs_ds = tuple(map(lambda n: self.xs_ds_all_data[n], self.xs_dataset_names))
            ys_ds = tuple(map(lambda n: self.ys_ds_all_data[n], self.ys_dataset_names))

        for i in range(num_xs):
            xs[i] = xs_ds[i][main_ix]
        for i in range(num_ys):
            ys[i] = ys_ds[i][main_ix]
        return xs, ys

    def read_data_point_augmented(self, main_ix, secondary_ix):
        """
        Read xs, ys for a particular index in the data augmented datasets
        :param main_ix: int. Index in the h5 file (main index)
        :secondary_ix: int. Index in the augmented dataset (main_ix - augmented_ix)
        :return: tuple of xs,ys in the dataset format
        """
        num_xs = len(self.xs_sizes)
        num_ys = len(self.ys_sizes)
        xs_augmented_ds = tuple(map(lambda n: self.h5[n], self.pregenerated_augmented_xs_ds_names))
        ys_augmented_ds = tuple(map(lambda n: self.h5[n], self.pregenerated_augmented_ys_ds_names))

        xs = [None] * num_xs
        ys = [None] * num_ys

        for i in range(num_xs):
            xs[i] = xs_augmented_ds[i][main_ix, secondary_ix]
        for i in range(num_ys):
            ys[i] = ys_augmented_ds[i][main_ix, secondary_ix]

        return xs, ys

    def get_next_batch(self, batch_size, batch_type):
        """
        Get a tuple for the next batch_size images and labels
        :param batch_size: int
        :param batch_type: int. One of the values in (H5Manager.TRAIN, H5Manager.VALIDATION, H5Manager.TEST)
        :return: Tuple of (batch_size x images, batch_size x labels)
        """
        assert batch_type in (self.TRAIN, self.VALIDATION, self.TEST), \
            "Wrong batch type. Choose one from H5Manager.TRAIN, H5Manager.VALIDATION, H5Manager.TEST"

        # Get the maximum number of data points that could be selected in this batch
        if batch_type == self.TRAIN:
            # remaining_data_points = self.num_train_points - self._train_ix_pos_
            if self.current_epoch == 0:
                # Initialize indexes
                self._new_epoch_()
            remaining_data_points = batch_size   # We will always return 'batch_size' elements
        # Otherwise we can return less elements than asked in the batch because we reach the end of the validation/test data
        elif batch_type == self.VALIDATION:
            if self._validation_ix_pos_ == 0 and self.use_pregenerated_augmented_val_data:
                if self._pregenerated_augmented_ixs_val_ is None:
                    # We need to create a list for each index in the training dataset.
                    # In order to loop over all the images in the dataset, we need main index
                    # that contains 'num_total_augmented_images' for each main image.
                    # We assume we have the same number of augmented data points for each used data dataset
                    self._pregenerated_augmented_ixs_val_ = np.tile(
                        np.arange(self.pregenerated_num_augmented_data_points_per_data_point),
                        self.num_validation_points).reshape(self.num_validation_points, -1)
                self._pregenerated_augmented_ixs_val_pos_ = [0] * self.num_validation_points

            if self.use_pregenerated_augmented_val_data:
                remaining_data_points = batch_size
            else:
                remaining_data_points = self.num_validation_points - self._validation_ix_pos_
        elif batch_type == self.TEST:
            remaining_data_points = self.num_test_points - self._test_ix_pos_
        else:
            raise Exception("Unknown batch type: {}".format(batch_type))

        num_data_points_current_batch = min(batch_size, remaining_data_points)

        num_xs = len(self.xs_sizes)
        num_ys = len(self.ys_sizes)
        batch_xs = [None] * num_xs
        batch_ys = [None] * num_ys
        for i in range(num_xs):
            batch_xs[i] = np.zeros((num_data_points_current_batch,) + self.xs_sizes[i], np.float32)
        for i in range(num_ys):
            batch_ys[i] = np.zeros((num_data_points_current_batch,) + self.ys_sizes[i], np.float32)

        batch_pos = 0
        while batch_pos < num_data_points_current_batch:
            if batch_type == self.TRAIN:
                with self.lock:
                    # Select the next main index
                    current_main_pos = self._train_ix_pos_
                    if self.load_all_data:
                        main_ix = self.train_ixs[current_main_pos:current_main_pos+num_data_points_current_batch]
                        self._train_ix_pos_ += num_data_points_current_batch
                    else:
                        main_ix = self.train_ixs[current_main_pos]
                        self._train_ix_pos_ += 1

                xs, ys = self.read_data_point(main_ix)

                for i in range(num_xs):
                    if self.load_all_data:
                        batch_xs[i][batch_pos:batch_pos+num_data_points_current_batch] = xs[i]
                    else:
                        batch_xs[i][batch_pos] = xs[i]
                for i in range(num_ys):
                    if self.load_all_data:
                        batch_ys[i][batch_pos:batch_pos+num_data_points_current_batch] = ys[i]
                    else:
                        batch_ys[i][batch_pos] = ys[i]

                if self.load_all_data:
                    batch_pos += num_data_points_current_batch
                else:
                    batch_pos += 1

                if self.use_pregenerated_augmented_train_data:
                    # Augmented images
                    aug = 0
                    while aug < self.num_augmented_train_data_points_per_data_point and batch_pos < num_data_points_current_batch:
                        # Pre-augmented dataset
                        secondary_ix = self._pregenerated_augmented_ixs_[current_main_pos][self._pregenerated_augmented_ixs_pos_[current_main_pos]]
                        self._pregenerated_augmented_ixs_pos_[current_main_pos] += 1

                        if self._pregenerated_augmented_ixs_pos_[current_main_pos] == self.pregenerated_num_augmented_data_points_per_data_point:
                            # Start over with the first augmented image for this original image
                            self._pregenerated_augmented_ixs_pos_[current_main_pos] = 0

                        # Read the data point from the dataset
                        augmented_xs, augmented_ys = self.read_data_point_augmented(main_ix, secondary_ix)

                        for i in range(num_xs):
                            batch_xs[i][batch_pos:batch_pos+num_data_points_current_batch] = augmented_xs[i]
                        for i in range(num_ys):
                            batch_ys[i][batch_pos:batch_pos+num_data_points_current_batch] = augmented_ys[i]
                        batch_pos += 1
                        aug += 1
                if self._train_ix_pos_ == self.num_train_points:
                    # We reached the end of a training dataset. End of epoch
                    self._new_epoch_()
            elif batch_type == self.VALIDATION:
                with self.lock:
                    # Select the next main index
                    current_main_pos = self._validation_ix_pos_
                    if self.load_all_data:
                        main_ix = self.validation_ixs[current_main_pos:current_main_pos+num_data_points_current_batch]
                        self._validation_ix_pos_ += num_data_points_current_batch
                    else:
                        main_ix = self.validation_ixs[current_main_pos]
                        self._validation_ix_pos_ += 1

                xs, ys = self.read_data_point(main_ix)
                for i in range(num_xs):
                    if self.load_all_data:
                        batch_xs[i][batch_pos:batch_pos+num_data_points_current_batch] = xs[i]
                    else:
                        batch_xs[i][batch_pos] = xs[i]
                for i in range(num_ys):
                    if self.load_all_data:
                        batch_ys[i][batch_pos:batch_pos+num_data_points_current_batch] = ys[i]
                    else:
                        batch_ys[i][batch_pos] = ys[i]

                if self.load_all_data:
                    batch_pos += num_data_points_current_batch
                else:
                    batch_pos += 1

                if self.use_pregenerated_augmented_val_data:
                    # Augmented images
                    aug = 0
                    while aug < self.num_augmented_train_data_points_per_data_point and batch_pos < num_data_points_current_batch:
                        # Pre-augmented dataset
                        val_secondary_ix = self._pregenerated_augmented_ixs_val_[current_main_pos][self._pregenerated_augmented_ixs_val_pos_[current_main_pos]]
                        self._pregenerated_augmented_ixs_val_pos_[current_main_pos] += 1

                        if self._pregenerated_augmented_ixs_val_pos_[current_main_pos] == self.pregenerated_num_augmented_data_points_per_data_point:
                            # Start over with the first augmented image for this original image
                            self._pregenerated_augmented_ixs_val_pos_[current_main_pos] = 0

                        # Read the data point from the dataset
                        augmented_xs, augmented_ys = self.read_data_point_augmented(main_ix, val_secondary_ix)

                        for i in range(num_xs):
                            batch_xs[i][batch_pos] = augmented_xs[i]
                        for i in range(num_ys):
                            batch_ys[i][batch_pos] = augmented_ys[i]
                        batch_pos += 1
                        aug += 1
                if self._validation_ix_pos_ == self.num_validation_points:
                    self._validation_ix_pos_ = 0
            else:
                # Test
                with self.lock:
                    # Select the next main index
                    current_main_pos = self._test_ix_pos_
                    if self.load_all_data:
                        main_ix = self.test_ixs[current_main_pos:current_main_pos+num_data_points_current_batch]
                        self._test_ix_pos_ += num_data_points_current_batch
                    else:
                        main_ix = self.test_ixs[current_main_pos]
                        self._test_ix_pos_ += 1

                    if self._test_ix_pos_ == self.num_test_points:
                      self._test_ix_pos_ = 0

                xs, ys = self.read_data_point(main_ix)
                for i in range(num_xs):
                    if self.load_all_data:
                        batch_xs[i][batch_pos:batch_pos+num_data_points_current_batch] = xs[i]
                    else:
                        batch_xs[i][batch_pos] = xs[i]
                for i in range(num_ys):
                    if self.load_all_data:
                        batch_ys[i][batch_pos:batch_pos+num_data_points_current_batch] = ys[i]
                    else:
                        batch_ys[i][batch_pos] = ys[i]

                if self.load_all_data:
                    batch_pos += num_data_points_current_batch
                else:
                    batch_pos += 1

        return batch_xs, batch_ys

    def get_steps_per_epoch_train(self):
        """
        Get the number of steps that will be needed to see all the original training data points (not having
        in mind the augmented)
        :return: int. Number of steps
        """
        return math.ceil(float(self.num_total_train_data_points_per_epoch / self.batch_size))

    def get_steps_validation(self):
        """
        Get the number of steps that will be needed to see all the validation data points
        :return: int
        """
        return math.ceil(float(self.num_validation_points / self.batch_size))

    def get_steps_test(self):
        """
        Get the number of steps that will be needed to see all the test data points
        :return: int
        """
        return math.ceil(float(self.num_test_points / self.batch_size))

    def reset_validation_index_pos(self):
        """
        Set the validation index position to the first element
        :return:
        """
        self._validation_ix_pos_ = 0


