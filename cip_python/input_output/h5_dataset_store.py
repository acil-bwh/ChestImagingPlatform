import os
import datetime
import h5py
import numpy as np
import traceback

class Axis(object):
    def __init__(self, name, description, size_, kind, units, splitted_axis_names=(), splitted_axis_descriptions=()):
        self.__axis_name__ = name
        self.__description__ = description
        self.__size__ = size_
        self.__kind__ = kind
        self.__units__ = units
        self.__splitted_axis_names__ = splitted_axis_names
        self.__splitted_axis_descriptions__ = splitted_axis_descriptions

        self.__validate__()

    @classmethod
    def new_index(cls, size_, name="index", description="Main index"):
        return cls(name, description, size_, H5DatasetStore.AXIS_IX, H5DatasetStore.UNIT_UNDEFINED)

    @property
    def name(self):
        return self.__axis_name__

    @property
    def description(self):
        return self.__description__

    @property
    def size_(self):
        return self.__size__

    @property
    def kind(self):
        return self.__kind__

    @property
    def units(self):
        return self.__units__

    @property
    def splitted_axis_names(self):
        return self.__splitted_axis_names__

    @property
    def splitted_axis_descriptions(self):
        return self.__splitted_axis_descriptions__


    def __validate__(self):
        assert self.__axis_name__, "An axis name is needed"
        assert self.__description__, "A description is needed"
        assert isinstance(self.__size__, int), "'size' should be an integer. Got: {}".format(type(self.__size__))
        assert self.__kind__ in (H5DatasetStore.AXIS_IX, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.AXIS_TIME,
                                 H5DatasetStore.AXIS_TABLE_IX, H5DatasetStore.UNIT_UNDEFINED)

        if not self.__class__.is_physical_axis(self.kind) and not self.kind == H5DatasetStore.AXIS_IX:
            # Check that each position in the axis has a name/description
            assert len(self.__splitted_axis_names__) == len(self.__splitted_axis_descriptions__) == self.__size__, \
                "I need the 'splitted_axis_names' and 'splitted_axis_description' tuples, please!"

    @classmethod
    def is_physical_axis(cls, kind):
        return kind in (H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.AXIS_TIME)



class H5DatasetStore(object):

    #####################################
    # CONSTANTS DEFINITION              #
    #####################################

    DS_NDARRAY = 'NDARRAY'
    DS_TABLE = 'TABLE'
    DS_KEY = 'KEY'
    DS_PROPERTY = 'METADATA'
    
    # Declare axis types
    AXIS_IX = 'INDEX'
    AXIS_SPATIAL = 'SPATIAL'
    AXIS_TIME = 'TIME'
    AXIS_TABLE_IX = 'AXIS_TABLE_IX'
    AXIS_UNDEFINED = 'UNDEFINED'

    # Declare some predefined units
    UNIT_PIXEL = 'pixel'
    UNIT_MM = 'mm'
    UNIT_SECOND = 'second'
    UNIT_UNDEFINED = 'undefined'

    #####################################
    # PUBLIC METHODS                    #
    #####################################
    
    def __init__(self, h5_file_name):
        self.h5_file_name = h5_file_name


    def create_h5_file(self, description, key_names=('sid', 'cid'),
                       keys_description="Element ids: subject id (sid) and case id (cid)",
                       override_if_existing=False):   
        """
        Create the H5 file from scratch. First method needed in the H5 generation
        :param description: str. H5 general description
        :param key_names: str-tuple. Names of the keys that will be used for each dataset in the H5
        :param keys_description: str. Brief description of the key names
        :param override_if_existing: bool. If True, the existence of a previous h5 file will be ignored
        :return: h5 File
        """
        if not override_if_existing and os.path.isfile(self.h5_file_name):
            raise Exception("File {} already exists".format(self.h5_file_name))
        
        with h5py.File(self.h5_file_name, 'w') as h5:
            h5.attrs['description'] = description
            h5.attrs['key_names'] = ';'.join(key_names)
            h5.attrs['keys_description'] = keys_description

    def create_ndarray(self, name, general_description, dtype, axes,
                       # shape, axes_description, axes_kind, axes_units,
                       enable_chunks=True, chunks=None, enable_max_shape=True):
        """
        Create a 'DATA' dataset
        :param name: str. Name of the dataset
        :param general_description: str. Dataset general description
        :param dtype: Numpy type. Dataset data type
        :param shape: int-tuple. Original dataset shape
        :param axes_description: str-tuple. Description for each axis in the data array. Ex: ('x','y','z')
        :param axes_kind: str-tuple. Kind for each axis. It must be one of the allowed AXIS_ values
        :param axes_units: str-tuple. Unit for each axis. See UNIT_XX properties for some predefined values
        :param axes_table_indexes: dictionary of str-lists. For each AXIS_TABLE_IX axis, set a list with the names 
                                   of the columns. Ex: [{3:['translation_x', 'translation_y'], 4:['noise_mean']}]
                                   
        :param enable_chunks: bool. Enable the dataset chunks
        :param chunks: int-tuple. Dataset chunks (typically one element unless otherwise specified)
        :param enable_max_shape: bool. Allow the dataset unlimited growth in the first axis
        :return: H5 Dataset
        """
        assert len(axes) > 0, "I needed a tuple/list of Axis objects!"
        assert axes[0].kind == self.AXIS_IX, "The first axis in the dataset must be '{}'".format(self.AXIS_IX)
        num_elems = axes[0].size_
        shape = tuple(ax.size_ for ax in axes)
        axes_kind = tuple(ax.kind for ax in axes)

        # Atomic dataset creation with corresponding metadata
        with h5py.File(self.h5_file_name, 'r+') as h5:
            ds = self.__create_dataset__(h5, self.DS_NDARRAY, name, general_description, dtype, shape,
                                         enable_chunks, chunks, enable_max_shape)
            # Save attributes
            ds.attrs['axes_name'] = ";".join(ax.name for ax in axes)
            ds.attrs['axes_description'] = ";".join(ax.description for ax in axes)
            ds.attrs['axes_kind'] = ";".join(ax.kind for ax in axes)
            ds.attrs['axes_units'] = ";".join(ax.units for ax in axes)

            # Create metadata datasets
            key_ds_name = name + ".key"
            ds.attrs['key_dataset'] = key_ds_name
            ds_keys = self.__create_dataset_key__(h5, key_ds_name, num_elems)

            physical_shape = self.get_spacing_shape(shape, axes_kind)
            spacing_ds_name = name + ".spacing"
            ds.attrs['spacing_dataset'] = spacing_ds_name
            ds_spacing = self.__create_dataset__(h5, self.DS_PROPERTY, spacing_ds_name, "spacing dataset", np.float32,
                                                 physical_shape, enable_chunks=False, enable_max_shape=enable_max_shape)
            ds_spacing.attrs['axes_name'] = ";".join("{}_spacing".format(ax.name) for ax in axes)

            origin_ds_name = name + ".origin"
            ds.attrs['origin_dataset'] = origin_ds_name
            ds_origin = self.__create_dataset__(h5, self.DS_PROPERTY, origin_ds_name,
                                                "origin dataset", np.float32, physical_shape,
                                                enable_chunks=False, enable_max_shape=enable_max_shape)
            ds_origin.attrs['axes_name'] = ";".join("{}_origin".format(ax.name) for ax in axes)

            missing_ds_name = name + ".missing"
            ds.attrs['missing_dataset'] = missing_ds_name
            ds_missing = self.__create_dataset__(h5, self.DS_PROPERTY, missing_ds_name, "missing flag",
                                                 np.uint8, (shape[0], 1), enable_chunks=False,
                                                 enable_max_shape=enable_max_shape)

            for ax in (ax for ax in axes if ax.kind == H5DatasetStore.AXIS_TABLE_IX):
                ds.attrs["table_ix_col_names_{}".format(ax.name)] = ";".join(ax.splitted_axis_names)
                ds.attrs["table_ix_col_descriptions_{}".format(ax.name)] = ";".join(ax.splitted_axis_descriptions)

    def create_table(self, name, general_description, columns_names, dtype, shape,
                     enable_chunks=False, chunks=None, enable_max_shape=True):
        """
        Create a 'TABLE' dataset
        :param name: str. Name of the dataset
        :param general_description: str. Dataset general description
        :param columns_names: str-tuple. Description for each column. Ex: ('spacing_x', 'spacing_y', 'spacing_z')
        :param dtype: Numpy type. Dataset data type
        :param shape: int-tuple. Original dataset shape
        :param enable_chunks: bool. Enable the dataset chunks
        :param chunks: int-tuple. Dataset chunks (typically one element unless otherwise specified)
        :param enable_max_shape: bool. Allow the dataset unlimited growth in the first axis
        :return: H5 Dataset
        """
        # Preconditions
        assert len(shape) == 2, "Only 2 dimensional arrays are allowed in this kind of dataset"

        with h5py.File(self.h5_file_name, 'r+') as h5:
            ds = self.__create_dataset__(h5, self.DS_TABLE, name, general_description, dtype, shape,
                                         enable_chunks=enable_chunks, chunks=chunks, enable_max_shape=enable_max_shape)
            ds.attrs['columns_names'] = ";".join(columns_names)

            key_ds_name = name + ".key"
            ds.attrs['key_dataset'] = key_ds_name
            ds_keys = self.__create_dataset_key__(h5, key_ds_name, shape[0])

    def insert_ndarray_single_point(self, ds_name, data_array, key_array,
                                    spacing_array, origin_array, missing):
        """
        Insert a single data point in a ndarray and all its associated datasets
        :param ds_name: str. Name of the dataset
        :param data_array: D1 X D2 X ... DN numpy array. Main data array (single point)
        :param key_array: num_keys numpy array. Array of strings with keys (sid, cid, etc.)
        :param S1 x S2 X ... X SN float array: Spacing metadata for each dimension in data_array
        :param origin_array: O1 x O2 X ... X ON float array. Origin metadata for each dimension in data_array
        :param missing: uint8. Missing data point
        :return int. Index of the last inserted element
        """
        return self.insert_ndarray_data_points(
            ds_name, np.expand_dims(data_array, 0),
            np.expand_dims(key_array, 0),
            np.expand_dims(spacing_array, 0),
            np.expand_dims(origin_array, 0),
            np.expand_dims(np.array([missing], np.uint8), 0)
            #np.array([1, missing], np.uint8)
        )

    def insert_ndarray_data_points(self, ds_name, data_array, key_array,
                                   spacing_array, origin_array, missing_array):
        """
        Insert several data points in a ndarray and all its associated datasets
        :param ds_name: str. Name of the dataset
        :param data_array: N x D1 X D2 X ... DN numpy array. Main data array
        :param key_array: N x num_keys numpy array. Array of strings with keys (sid, cid, etc.)
        :param N x S1 x S2 X ... X SN float array: Spacing metadata for each dimension in data_array
        :param origin_array: N x O1 x O2 X ... X ON float array. Origin metadata for each dimension in data_array
        :param missing_array: N x M1 x ... x MN uint8 array. Missing metadata for each index
        :return int. Index of the last inserted element
        """
        # Check that input array shapes align according to convention
        with h5py.File(self.h5_file_name, 'r+') as h5:
            num_elems = data_array.shape[0]
            assert key_array.shape[0] == spacing_array.shape[0] == origin_array.shape[0] == missing_array.shape[0]  \
                   == num_elems, \
                    "Datasets do not align in the first dimension"
            # Read the dataset axes information
            axes_kind = h5[ds_name].attrs['axes_kind'].split(";")
            shape = list(self.get_spacing_shape(h5[ds_name].shape, axes_kind))
            shape[0] = num_elems
            shape = tuple(shape)
            if spacing_array is not None:
                assert spacing_array.shape == origin_array.shape == shape, \
                    "Spacing and origin do not contain correct number of components to match data array axes"
            else:
                # Make sure that there are not any physical axes
                assert shape[1] == 0, "Missing spacing information for some axis"

            assert missing_array.shape[1] == 1, "missing_array shape[1] is not 1"

            metadata_elems = ('key', 'spacing', 'origin', 'missing')

            # Atomic insertion

            # Make sure that the indexes for all the datasets are aligned
            last_ix = h5[ds_name].attrs['last_ix']
            for metadata in metadata_elems:
                # Get the name of the dataset that contains the corresponding metadata for this NDARRAY dataset
                md_ds_name = h5[ds_name].attrs[metadata + '_dataset']
                assert h5[md_ds_name].attrs['last_ix'] == last_ix, "The dataset and its metadata are not aligned"

            assert key_array.shape[1] == len(self.get_key_names()), "Key shape does not match"

            first_ix = last_ix + 1
            last_ix = first_ix + num_elems - 1
            assert last_ix < h5[ds_name].shape[0], "There is no space in the dataset to insert all the points"

            # Insert all the data
            self.__validate_ndarray_structure__(h5, ds_name)
            inserted = []
            try:
                h5[ds_name][first_ix:last_ix + 1] = data_array
                h5[ds_name].attrs['last_ix'] = last_ix
                inserted.append("data")

                # Insert metadata
                meta_data = dict()
                meta_data['key'] = key_array
                meta_data['missing'] = missing_array
                if spacing_array is not None:
                    meta_data['spacing'] = spacing_array
                    meta_data['origin'] = origin_array

                for md in metadata_elems:
                    md_ds_name = h5[ds_name].attrs[md + '_dataset']
                    h5[md_ds_name][first_ix:last_ix + 1] = meta_data[md]
                    h5[md_ds_name].attrs['last_ix'] = last_ix
                    inserted.append(md)
            except Exception as ex:
                print("There was an error while inserting the data points. "
                      "The following data were inserted succesfully at indexes {}-{}: {}".format(first_ix, last_ix,
                                                                                                 inserted))
                raise ex
        return last_ix

    def insert_table_single_point(self, ds_name, data_array, key_array):
        """
        Insert a single point in a TABLE object
        :param ds_name: str. Name of the dataset
        :param data_array: Numpy array. Data to insert
        :param key_array: Numpy array of strings. Keys associated to the data
        :return: int. Last index of the inserted elements
        """
        return self.insert_table_data_points(ds_name, np.expand_dims(data_array, 0),
                                             np.expand_dims(key_array, 0))

    def insert_table_data_points(self, ds_name, data_array, key_array):
        """
        Insert multiple points in a TABLE object
        :param ds_name: str. Name of the dataset
        :param data_array: Numpy array. Data to insert
        :param key_array: Numpy array of strings. Keys associated to the data
        :return: int. Last index of the inserted elements
        """
        # Make sure the keys are correct
        # Check that input array shapes align according to convention
        num_elems = data_array.shape[0]

        assert key_array.shape[0] == num_elems, "Datasets do not align in the first dimension"
        assert len(data_array.shape) == 2, "Only 2D data are allowed for tables"

        # Atomic insertion
        with h5py.File(self.h5_file_name, 'r+') as h5:
            self.__validate_keys_structure__(h5, ds_name)

            # Make sure that the key dataset is aligned
            assert key_array.shape[1] == len(self.get_key_names()), "Key shape does not match"

            last_ix = h5[ds_name].attrs['last_ix']
            ds_keys_name = h5[ds_name].attrs['key_dataset']
            assert h5[ds_keys_name].attrs['last_ix'] == last_ix, "The dataset and its keys are not aligned"

            first_index = last_ix + 1
            last_ix = first_index + num_elems - 1

            assert last_ix < h5[ds_name].shape[0], "There is no space in the dataset to insert all the points"

            # Insert the data
            h5[ds_name][first_index:last_ix + 1] = data_array
            h5[ds_name].attrs['last_ix'] = last_ix

            # Insert keys
            keys_ds_name = h5[ds_name].attrs['key_dataset']
            h5[keys_ds_name][first_index:last_ix + 1] = key_array
            h5[keys_ds_name].attrs['last_ix'] = last_ix

        return last_ix



    def get_spacing_shape(self, shape, axes_kind):
        """
        Get two tuples with shape and axes kind and calculate the shape of the physical axes that will be needed to
        store spacing, origin, etc.
        :param shape: int-tuple. Shape of the dataset (first dimension=number of elements)
        :param axes_kind: tuple of str. Axes kind
        :return: tuple. Shape needed
        """
        physical_axes = [ax for ax in axes_kind if ax in (self.AXIS_SPATIAL, self.AXIS_TIME)]
        num_elems = shape[0]
        num_physical = len(physical_axes)
        s = [num_elems, num_physical]
        for i in range(1, len(shape)):
            if axes_kind[i] not in (self.AXIS_SPATIAL, self.AXIS_TIME):
                s.append(shape[i])

        return tuple(s)

    def get_main_dataset_keys(self):
        """
        Get a list with the names of the main datasets in the file
        """
        with h5py.File(self.h5_file_name, 'r') as h5:
            return [dskey for dskey in h5 if h5[dskey].attrs['kind'] in (self.DS_NDARRAY, self.DS_TABLE)]


    def get_key_names(self):
        """
        Get a tuple of key names that will be used in each dataset
        :return: str-tuple
        """
        with h5py.File(self.h5_file_name, 'r') as h5:
            return tuple(h5.attrs['key_names'].split(';'))
    
    def get_keys_description(self):
        """
        Keys description (str)
        :return: str
        """
        with h5py.File(self.h5_file_name, 'r') as h5:
            return h5.attrs['keys_description']


    def full_validation(self, include_keys_analysis=False):
        """
        Check that all the conventions are followed in the dataset.
        The method will raise an assertion error if something goes wrong
        :param include_keys_analysis: bool. When True, not only the schema will be validated, but the method asserts that
                                      all the keys datasets match exactly
        """
        main_keys = self.get_main_dataset_keys()
        assert len(main_keys) > 0, "Empty dataset"
        with h5py.File(self.h5_file_name, 'r') as h5:
            num_elems = h5[main_keys[0]].shape[0]
            last_ix = h5[main_keys[0]].attrs['last_ix']
            # Validate schema and number of elements
            for key in main_keys:
                if h5[key].attrs['kind'] == self.DS_NDARRAY:
                    self.__validate_ndarray_structure__(h5, key)
                else:
                    self.__validate_keys_structure__(h5, key)
                assert h5[key].shape[0] == num_elems, "Wrong number of elements in dataset {}. Expected: {}. Got: {}" \
                                                    .format(key, num_elems, h5[key].shape[0])
                assert h5[key].attrs['last_ix'] == last_ix, "Wrong value of 'last_ix' attribute in dataset {}. Expected: {}. Got: {}" \
                                                    .format(key, last_ix, h5[key].attrs['last_ix'])

            if include_keys_analysis and len(main_keys) > 1:
                # Make sure that all the keys are aligned
                ds_keys = [key for key in h5 if 'kind' in h5[key].attrs and h5[key].attrs['kind'] == self.DS_KEY]
                # Preload keys for efficiency
                k0 = h5[ds_keys[0]][:]
                for j in range(1, len(ds_keys)):
                    k1 = h5[ds_keys[j]][:]
                    assert np.array_equal(k0, k1), "I found some keys misalignment in dataset {}!".format(ds_keys[j])


    def concat_full_h5s(self, h5_file_paths):
        """
        Concat a series of h5 files fully filled (same schema) and concat them in a single H5
        :param h5_file_paths: list of string. Paths to the h5 files
        :return: list of errors
        """
        assert len(h5_file_paths) >= 2, "The list must contain at least 2 elements"
        errors = []

        # Create the initial collection of datasets
        with h5py.File(self.h5_file_name, 'r+') as h5f:
            h5s_small = H5DatasetStore(h5_file_paths[0])
            # Validate the original dataset
            h5s_small.full_validation()
            # Clone all the datasets
            timestamp = datetime.datetime.now().isoformat()
            with h5py.File(h5_file_paths[0], 'r') as smh5f:
                for key in smh5f:
                    ds = smh5f[key]
                    ds_copy = h5f.create_dataset(key, data=ds[:], maxshape=(None,) + ds.shape[1:], chunks=ds.chunks)
                    for attr in (attr for attr in ds.attrs if attr != "timestamp"):
                        ds_copy.attrs[attr] = ds.attrs[attr]
                    ds_copy.attrs['timestamp'] = timestamp

            # Iterate over the rest of the datasets
            for i in range(1, len(h5_file_paths)):
                try:
                    h5s_small = H5DatasetStore(h5_file_paths[i])
                    # Validate the original dataset
                    h5s_small.full_validation()
                    # Make sure there are the same datasets
                    with h5py.File(h5_file_paths[i], 'r') as smh5f:
                        if set(h5f.keys()) != set(smh5f.keys()):
                            raise Exception("'{}' has a different number of datasets".format(h5_file_paths[i]))
                        for key in smh5f:
                            ds_small = smh5f[key]
                            ds_big = h5f[key]
                            num_points_small = ds_small.shape[0]
                            # Use the 'last_ix' attribute to overwrite possible failed cases
                            last_ix = ds_big.attrs['last_ix']
                            # Resize dataset
                            ds_big.resize((last_ix + num_points_small + 1,) + ds_small.shape[1:])
                            # Copy data
                            ds_big[last_ix + 1 : last_ix + 1 + num_points_small] = ds_small[:]

                        # Update last_ix attribute to keep track of the inserted data points in an atomic way
                        for key in h5f:
                            h5f[key].attrs['last_ix'] += num_points_small
                except:
                    # Register the exception
                    errors.append((h5_file_paths[i], traceback.format_exc()))
            return errors


    def concat_h5_datasets(self, h5_file_paths):
        """
        Concat a series of h5 files where each file contains one or more dataset objects, and store
        them in a single H5 object
        :param h5_file_paths: list of string. Paths to the h5 files
        """
        assert len(h5_file_paths) >= 2, "The list must contain at least 2 elements"
        errors = []

        with h5py.File(self.h5_file_name, 'r+') as h5f:
            timestamp = datetime.datetime.now().isoformat()
            for p in h5_file_paths:
                try:
                    h5s_small = H5DatasetStore(p)
                    # Validate the original dataset
                    h5s_small.full_validation()
                    # Clone all the datasets
                    with h5py.File(p, 'r') as smh5f:
                        for key in smh5f:
                            ds = smh5f[key]
                            # Copy data
                            ds_copy = h5f.create_dataset(key, data=ds[:], maxshape=ds.maxshape, chunks=ds.chunks)
                            # Copy all the attributes (except timestamp)
                            for attr in (attr for attr in ds.attrs if attr != "timestamp"):
                                ds_copy.attrs[attr] = ds.attrs[attr]
                            # Set the same timestamp attribute for all the datasets
                            ds_copy.attrs['timestamp'] = timestamp
                except:
                    # Register the exception
                    errors.append((p, traceback.format_exc()))

        # Final validation
        self.full_validation()
        return errors

    #####################################
    # PRIVATE METHODS                   #
    #####################################

    def __create_dataset__(self, h5, ds_kind, name, description, dtype, shape,
                           enable_chunks=True, chunks=None, enable_max_shape=True):
        """
        Create a dataset in the H5File. This method should not be called directly (use the public methods instead)
        :param h5: h5 File handler
        :param ds_kind: str. Kind of dataset ('DS_NDARRAY' | 'DS_TABLE' | 'DS_KEY' | 'DS_PROPERTY').
        :param name: str. Dataset name
        :param description: str. Dataset 'description' attribute
        :param dtype: numpy type. Dataset data type
        :param shape: int-tuple. Original dataset shape (first dimension=number of data points).
        :param enable_chunks: bool. Set dataset chunks (typically one data point unless 'chunks' argument contains a value)
        :param chunks: int-tuple. Dataset chunks (typically one element unless otherwise specified)
        :param enable_max_shape: bool. Allow the dataset unlimited growth (in the first dimension)
        :return: H5 Dataset
        """
        if not enable_chunks:
            chunks = None
        elif chunks is None:
            # Default chunks
            chunks = (1,) + shape[1:]
        if enable_max_shape:
            ds = h5.create_dataset(name, dtype=dtype, shape=shape, chunks=chunks, maxshape=(None,)+shape[1:])
        else:
            ds = h5.create_dataset(name, dtype=dtype, shape=shape, chunks=chunks)

        ds.attrs['kind'] = ds_kind
        ds.attrs['description'] = description
        ds.attrs['timestamp'] = datetime.datetime.now().isoformat()
        ds.attrs['last_ix'] = -1

        return ds

    def __create_dataset_key__(self, h5, name, num_elems, enable_chunks=False, chunks=None, enable_max_shape=True):
        """
        Create a dataset of kind 'DS_KEY'. Example: case ids, subject ids, etc.
        :param h5: h5 file object
        :param name: str.
        :param num_elems: int. Number of elements
        :param enable_chunks: bool. Enable the dataset chunks
        :param chunks: int-tuple. Dataset chunks (typically one element unless otherwise specified)
        :param enable_max_shape: bool. Allow unlimited growth
        :return: H5 Dataset
        """

        dt = h5py.special_dtype(vlen=str)
        key_names = self.get_key_names()
        n_keys = len(key_names)
        ds = self.__create_dataset__(h5, self.DS_KEY, name, self.get_keys_description(), dt, (num_elems, n_keys),
                                     enable_chunks=enable_chunks, chunks=chunks, enable_max_shape=enable_max_shape)
        return ds

    def __validate_ndarray_structure__(self, h5, ds_name):
        """
        Make all the necessary schema comprobations for a NDARRAY.
        If everything goes well, the method will return nothing. Otherwise, an assertion error will be raised
        :param h5: H5 File
        :param ds_name: name of the dataset
        """
        assert ds_name in h5, "Dataset {} not found in the H5 file".format(ds_name)
        # check kind
        assert 'kind' in h5[ds_name].attrs and h5[ds_name].attrs[
            'kind'] == self.DS_NDARRAY, "The dasaset has the wrong type"

        self.__validate_keys_structure__(h5, ds_name)
        # check metadata arrays
        for md, kind in zip(('spacing', 'origin', 'missing'), (self.DS_PROPERTY, self.DS_PROPERTY, self.DS_PROPERTY)):
            n = md + '_dataset'
            assert n in h5[ds_name].attrs, "Attribute '{}' could not be found in the dataset".format(n)
            md_ds_name = h5[ds_name].attrs[n]
            assert md_ds_name in h5, "Dataset {} not found".format(md_ds_name)
            assert 'kind' in h5[md_ds_name].attrs, "Attribute 'kind' not found in {}".format(md_ds_name)
            assert h5[md_ds_name].attrs[
                       'kind'] == kind, "Dataset '{}' was expected to be '{}' but it is '{}' instead".format(
                md_ds_name, kind, h5[md_ds_name].attrs['kind'])

    def __validate_keys_structure__(self, h5, ds_name):
        """
        Check the keys structure for the specified dataset
        :param h5: h5 File object
        :param ds_name: str. Name of the dataset
        """
        n = 'key_dataset'
        kind = self.DS_KEY
        assert n in h5[ds_name].attrs, "Attribute '{}' could not be found in the dataset".format(n)
        md_ds_name = h5[ds_name].attrs[n]
        assert md_ds_name in h5, "Dataset {} not found".format(md_ds_name)
        assert 'kind' in h5[md_ds_name].attrs, "Attribute 'kind' not found in {}".format(md_ds_name)
        assert h5[md_ds_name].attrs['kind'] == kind, "Dataset '{}' was expected to be '{}' but it is '{}' instead" \
            .format(md_ds_name, kind, h5[md_ds_name].attrs['kind'])

import unittest
import tempfile
class H5DatasetStoreTests(unittest.TestCase):
    def test_create_simple(self):
        """
        Create simple datasets
        """
        folder = tempfile.mkdtemp()
        os.makedirs(folder, exist_ok=True)
        output_h5_file = os.path.join(folder, "test.h5")

        h5 = H5DatasetStore(output_h5_file)
        # Keys: (sid, cid)
        h5.create_h5_file("This is a test description", override_if_existing=True)

        axes = [
            Axis.new_index(3),
            Axis("x", "My X", 10, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
            Axis("y", "My Y", 10, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
        ]
        h5.create_ndarray('images', 'images of something', np.int16, axes)
        h5.create_table('labels', 'table with categories', ('class1', 'class2'), np.int8, (3, 2))

        image_array = np.zeros((2, 10, 10), dtype=np.int16)
        spacing_array = np.zeros((2, 2), dtype=np.float32)
        origin_array = np.zeros((2, 2), dtype=np.float32)
        keys_array = np.array([
            ['1000X2', '1000X2_INSP_STD_COPD'],
            ['1000X2', '1000X2_EXP_STD_COPD']])
        missing_array = np.zeros((2, 1), dtype=np.uint8)

        # Multipoint insertion array
        ix = h5.insert_ndarray_data_points('images', image_array, keys_array, spacing_array, origin_array,
                                           missing_array)
        assert ix == 1

        # Single point insertion array
        ix = h5.insert_ndarray_single_point('images', image_array[0], keys_array[0], spacing_array[0], origin_array[0], 0)
        assert ix == 2

        # Multipoint insertion table
        table_array = np.array([[1, 0], [0, 1]], dtype=np.int8)
        ix = h5.insert_table_data_points('labels', table_array, keys_array)
        assert ix == 1

        # Single point insertion table
        ix = h5.insert_table_single_point('labels', table_array[0], keys_array[0])

        axes = [
            Axis.new_index(5),
            Axis.new_index(10, "secondary", "secondary index"),     # Augmented images
            Axis("x", "My X", 3, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
            Axis("y", "My Y", 3, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
        ]
        augmented = np.zeros((10, 3, 3), dtype=np.int16)
        h5.create_ndarray('augmented_indexes', 'images of something', np.int16, axes)
        spacing_array = np.random.random((2, 10))
        origin_array = np.random.random((2, 10))
        h5.insert_ndarray_single_point('augmented_indexes', augmented, keys_array[0], spacing_array, origin_array, 1)

        # Manipulate index manually
        with h5py.File(output_h5_file, 'r+') as h5f:
            h5f['labels.key'].attrs['last_ix'] += 1
        try:
            # This sentence should fail
            ix = h5.insert_table_data_points('labels', table_array, keys_array)
        except AssertionError:
            print("Nice catch!")
            pass

    def test_vertical_concat(self):
        """
        Concatenate several h5 files with the same schema ("vertical concat")
        """
        folder = tempfile.mkdtemp()
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, "ds1.h5")
        h5 = H5DatasetStore(p)
        # Keys: (sid, cid)
        h5.create_h5_file("This is a test description", override_if_existing=True)
        # 1. Create from scratch
        axes = [
            Axis.new_index(3),
            Axis("x", "My X", 10, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
            Axis("y", "My Y", 10, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
        ]
        h5.create_ndarray('images', 'images of something', np.int16, axes)

        h5.create_table('labels', 'table with categories', ('class1', 'class2'), np.int8, (3, 2))

        image_array = np.random.randint(0, 10, (2, 10, 10), dtype=np.int16)
        spacing_array = np.random.random((2, 2))
        origin_array = np.random.random((2, 2))
        keys_array = np.array([
            ['1000X2', '1000X2_INSP_STD_COPD'],
            ['1000X2', '1000X2_EXP_STD_COPD']])
        missing_array = np.zeros((2, 1), dtype=np.uint8)

        # Multipoint insertion array
        ix = h5.insert_ndarray_data_points('images', image_array, keys_array, spacing_array, origin_array, missing_array)
        assert ix == 1

        # Single point insertion array
        ix = h5.insert_ndarray_single_point('images', image_array[0], keys_array[0], spacing_array[0], origin_array[0],
                                            missing_array[0])
        assert ix == 2

        # Multipoint insertion table
        table_array = np.array([[1, 0], [0, 1]], dtype=np.int8)
        ix = h5.insert_table_data_points('labels', table_array, keys_array)
        assert ix == 1

        # Single point insertion table
        ix = h5.insert_table_single_point('labels', table_array[0], keys_array[0])

        # "Duplicate" h5
        ds2 = H5DatasetStore(os.path.join(folder, "ds2.h5"))
        ds2.create_h5_file("Copy dataset", override_if_existing=True)
        errors = ds2.concat_full_h5s([p,p])

        assert len(errors) == 0

        p2 = os.path.join(folder, "copy.h5")
        import shutil
        shutil.copy(p, p2)
        with h5py.File(p2) as h5:
            del(h5['labels'])
        with h5py.File(p2) as h5:
            h5.create_dataset('labels', (2,), dtype=np.bool)

        ds2 = H5DatasetStore(os.path.join(folder, "ds2.h5"))
        ds2.create_h5_file("Copy dataset", override_if_existing=True)
        errors = ds2.concat_full_h5s([p, p2, p])

        assert len(errors) > 0
        # import pprint
        # pprint.pprint(errors)

    def test_horizontal_concat(self):
        """
        Concatenate several h5 files with different datasets ("horizontal concat")
        """
        folder = tempfile.mkdtemp()
        os.makedirs(folder, exist_ok=True)
        # Dataset 1
        p1 = os.path.join(folder, "ds1.h5")
        h5 = H5DatasetStore(p1)
        # Keys: (sid, cid)
        h5.create_h5_file("This is a test description", override_if_existing=True)

        axes = [
            Axis.new_index(3),
            Axis("x", "My X", 10, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
            Axis("y", "My Y", 10, H5DatasetStore.AXIS_SPATIAL, H5DatasetStore.UNIT_PIXEL),
        ]
        h5.create_ndarray('images', 'images of something', np.int16, axes)

        image_array = np.random.randint(0, 10, (2, 10, 10), dtype=np.int16)
        keys_array = np.array([
            ['1000X2', '1000X2_INSP_STD_COPD'],
            ['1000X2', '1000X2_EXP_STD_COPD']])
        missing_array = np.zeros((2, 1), dtype=np.uint8)
        spacing_array = np.random.random((2, 2))
        origin_array = np.random.random((2, 2))

        ix = h5.insert_ndarray_data_points('images', image_array, keys_array, spacing_array, origin_array, missing_array)
        assert ix == 1

        # Dataset 2
        p2 = os.path.join(folder, "ds2.h5")
        h5 = H5DatasetStore(p2)
        # Keys: (sid, cid)
        h5.create_h5_file("This is a test description", override_if_existing=True)
        h5.create_table('labels', 'table with categories', ('class1', 'class2'), np.int8, (3, 2))

        # Multipoint insertion table
        table_array = np.array([[1, 0], [0, 1]], dtype=np.int8)
        ix = h5.insert_table_data_points('labels', table_array, keys_array)
        assert ix == 1

        # Dataset 3 (misaligned)
        p3 = os.path.join(folder, "ds3.h5")
        h5 = H5DatasetStore(p3)
        # Keys: (sid, cid)
        h5.create_h5_file("This is a test description", override_if_existing=True)
        h5.create_table('labels_2', 'table with categories', ('class1', 'class2'), np.int8, (2, 2))
        # Single insertion
        ix = h5.insert_table_single_point('labels_2', table_array[0], keys_array[0])

        # Create new datasets
        p = os.path.join(folder, "ds.h5")
        h5 = H5DatasetStore(p)
        h5.create_h5_file("This is a test description", override_if_existing=True)
        errors = h5.concat_h5_datasets([p1, p2])
        assert len(errors) == 0
        h5.full_validation(include_keys_analysis=True)

        h5 = H5DatasetStore(p)
        h5.create_h5_file("This is a test description", override_if_existing=True)
        catched = False
        try:
            h5.concat_h5_datasets([p1, p2, p3])
        except AssertionError as ex:
            print(ex)
            print("Nice catch!")
            catched = True
        assert catched, "It should have failed!"

if __name__ == "__main__":
    unittest.main()


