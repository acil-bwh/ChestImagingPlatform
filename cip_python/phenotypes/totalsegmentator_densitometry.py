import numpy as np
from scipy.stats import mode, kurtosis, skew
from argparse import ArgumentParser
import warnings
import platform
import datetime
import pandas as pd

from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter


class_map = {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "aorta",
        8: "inferior_vena_cava",
        9: "portal_vein_and_splenic_vein",
        10: "pancreas",
        11: "adrenal_gland_right",
        12: "adrenal_gland_left",
        13: "lung_upper_lobe_left",
        14: "lung_lower_lobe_left",
        15: "lung_upper_lobe_right",
        16: "lung_middle_lobe_right",
        17: "lung_lower_lobe_right",
        18: "vertebrae_L5",
        19: "vertebrae_L4",
        20: "vertebrae_L3",
        21: "vertebrae_L2",
        22: "vertebrae_L1",
        23: "vertebrae_T12",
        24: "vertebrae_T11",
        25: "vertebrae_T10",
        26: "vertebrae_T9",
        27: "vertebrae_T8",
        28: "vertebrae_T7",
        29: "vertebrae_T6",
        30: "vertebrae_T5",
        31: "vertebrae_T4",
        32: "vertebrae_T3",
        33: "vertebrae_T2",
        34: "vertebrae_T1",
        35: "vertebrae_C7",
        36: "vertebrae_C6",
        37: "vertebrae_C5",
        38: "vertebrae_C4",
        39: "vertebrae_C3",
        40: "vertebrae_C2",
        41: "vertebrae_C1",
        42: "esophagus",
        43: "trachea",
        44: "heart_myocardium",
        45: "heart_atrium_left",
        46: "heart_ventricle_left",
        47: "heart_atrium_right",
        48: "heart_ventricle_right",
        49: "pulmonary_artery",
        50: "brain",
        51: "iliac_artery_left",
        52: "iliac_artery_right",
        53: "iliac_vena_left",
        54: "iliac_vena_right",
        55: "small_bowel",
        56: "duodenum",
        57: "colon",
        58: "rib_left_1",
        59: "rib_left_2",
        60: "rib_left_3",
        61: "rib_left_4",
        62: "rib_left_5",
        63: "rib_left_6",
        64: "rib_left_7",
        65: "rib_left_8",
        66: "rib_left_9",
        67: "rib_left_10",
        68: "rib_left_11",
        69: "rib_left_12",
        70: "rib_right_1",
        71: "rib_right_2",
        72: "rib_right_3",
        73: "rib_right_4",
        74: "rib_right_5",
        75: "rib_right_6",
        76: "rib_right_7",
        77: "rib_right_8",
        78: "rib_right_9",
        79: "rib_right_10",
        80: "rib_right_11",
        81: "rib_right_12",
        82: "humerus_left",
        83: "humerus_right",
        84: "scapula_left",
        85: "scapula_right",
        86: "clavicula_left",
        87: "clavicula_right",
        88: "femur_left",
        89: "femur_right",
        90: "hip_left",
        91: "hip_right",
        92: "sacrum",
        93: "face",
        94: "gluteus_maximus_left",
        95: "gluteus_maximus_right",
        96: "gluteus_medius_left",
        97: "gluteus_medius_right",
        98: "gluteus_minimus_left",
        99: "gluteus_minimus_right",
        100: "autochthon_left",
        101: "autochthon_right",
        102: "iliopsoas_left",
        103: "iliopsoas_right",
        104: "urinary_bladder"
    }


class Phenotypes:
    """Base class for phenotype genearting classes.
    
    Attributes
    ----------
    pheno_names_ : list of strings
        The names of the phenotypes

    key_names_ : list of strings
        The names of the keys that are associated with each of the phenotype
        values.

    valid_key_values_ : dictionary
        The keys of the dictionary are the key names. Each dictionary key
        maps to a list of valid DB key values that each key name can
        assume

    Notes
    -----
    This class is meant to be inherited from in order to compute actual
    phenotype values. Subclasses are meant to:
    1) Overload the 'declare_pheno_names' method which should return a
    list of strings indicating the phenotype names
    2) Overload the 'execute' method which is expected to return the pandas
    dataframe, self._df
    3) Overload 'get_cid' in order to return a string value indicating the
    case ID (CID) associated with the case on which the phenotypes are
    executed.
    4) It is expected that within an inheriting classes 'execute' implementation
    the calls will be made to the 'add_pheno' method, which is implemented in
    this base class. 'add_pheno' handles all the heavy lifting to make sure that
    the dataframe is properly updated. After all desired calls to 'add_pheno'
    have been made, simply returning self._df should be sufficient for the
    class to behave properly.
    """
    def __init__(self):
        """
        """
        self.pheno_names_ = self.declare_pheno_names()
        self.key_names_ = self.declare_key_names()
        self.valid_key_values_ = self.valid_key_values()

        self.static_names_handler_ = {'Version': self.get_version,
                                      'Machine': self.get_machine,
                                      'OS_Name': self.get_os_name,
                                      'OS_Version': self.get_os_version,
                                      'OS_Kernel': self.get_os_kernel,
                                      'OS_Arch': self.get_os_arch,
                                      'Run_TimeStamp': self.get_run_time_stamp,
                                      'Generator': self.get_generator,
                                      'CID': self.get_cid}

        # Intializes the dataframe with static column names, key names, and
        # phenotype names.
        cols = list(self.static_names_handler_.keys())
        for n in self.key_names_:
            cols.append(n)

        for p in self.pheno_names_:
            cols.append(p)
    
        self._df = pd.DataFrame(columns=cols)    
    
    def get_generator(self):
        """Get the name of phenotype class that is producing the phenotype data

        Returns
        -------
        generator : string
            Name of phenotype class that is producing the phenotype data
        """
        return self.__class__.__name__

    def get_version(self):
        """Get the code version used to generate the phenotype data

        Returns
        -------
        version : string
            The git commit version used to generate the phenotype data

        TODO
        ----
        Need to figure out how to obtain this!
        """
        return 'NaN'

    def get_run_time_stamp(self):
        """Get the run-time stamp.

        Returns
        -------
        run_time_stamp : string
            The run-time stamp
        """
        return datetime.datetime.now().isoformat()

    def get_os_arch(self):
        """Get the OS architecture on which the phenotype data is generated.

        Returns
        -------
        os_arch : string
            The OS architecture on which the phenotype data is generated.
        """
        return platform.uname()[4]

    def get_os_kernel(self):
        """Get the OS kernel on which the phenotype data is generated.
    
        Returns
        -------
        os_kernel : string
            The OS kernel on which the phenotype data is generated.
        """
        return platform.uname()[2]

    def get_os_version(self):
        """Get the OS version on which the phenotype data is generated.
    
        Returns
        -------
        os_version : string
            The OS version on which the phenotype data is generated.
        """
        return platform.uname()[3]

    def get_os_name(self):
        """Get the OS name on which the phenotype data is generated.
    
        Returns
        -------
        os_name : string
            The OS name on which the phenotype data is generated.
        """
        return platform.uname()[0]

    def get_machine(self):
        """Get the machine name on which the phenotype data is generated.
    
        Returns
        -------
        machine : string
            The machine name on which the phenotype data is generated.
        """
        return platform.uname()[1]

    def declare_pheno_names(self):
        """
        """
        pass

    def declare_key_names(self):
        """
        """
        return ['Region', 'Type']

    def valid_key_values(self):
        """Get the valid DB key values that each DB key can assume.
    
        Returns
        -------
        valid_values : dictionary
            The returned dictionary keys are the valid DB key names. Each dictionary
            key value maps to a list of valid names that the DB key can assume.
        """
        
        valid_values = dict()
    
        region_names = ["WildCard"]
        for i in range(1, 105):
            region_names.append(class_map[i])
        valid_values['Region'] = region_names

        type_names = ["WildCard"]
        for i in range(1,105):
            type_names.append("WildCard")
        valid_values['Type'] = type_names
        
        return valid_values
  
    def get_cid(self):
        """To be implemented by inheriting classes
        """
        return None

    def add_pheno(self, key_value, pheno_name, pheno_value):
        """Add a phenotype.
    
        Parameters
        ----------
        key_value : list of strings
            This list indicates the specific DB key values that will be
            associated with the phenotype. E.g. ['WHOLELUNG', 'UNDEFINEDTYPE']

        pheno_name : string
            The name of the phenotype. E.g. 'LAA-950'

        pheno_value : object
            The phenotype value. Can be a numerical value, string, etc
        """



        num_keys = len(key_value)
    
        # Make sure CID has been set
        assert self.static_names_handler_['CID'] is not None, \
            "CID has not been set"
        
        # Check if pheno_name is valid
        assert pheno_name in self.pheno_names_, \
          "Invalid phenotype name: %s" % pheno_name
        
        # Check if key is valid
        for i in range(0, num_keys):
            assert key_value[i] in self.valid_key_values_[self.key_names_[i]], \
                "Invalid key: %s" % key_value[i]
          
        # Check if key already exists, otherwise add entry to data frame
        key_exists = True
        key_row = np.ones(len(self._df.index), dtype=bool)
        for i in range(0, len(self.key_names_)):
            key_row = \
                np.logical_and(key_row, \
                               self._df[self.key_names_[i]] == key_value[i])


        if np.sum(key_row) == 0:
            tmp = dict()
            for k in self.static_names_handler_.keys():
                tmp[k] = self.static_names_handler_[k]()
            tmp[pheno_name] = pheno_value
            for i in range(0, num_keys):
                tmp[self.key_names_[i]] = key_value[i]
            self._df = self._df.append(tmp, ignore_index=True)
        else:
            self._df.loc[np.where(key_row==True)[0][0],pheno_name] = pheno_value
        
    def to_csv(self, filename):
        self._df.to_csv(filename, index=False)
  
    def execute(self):
        pass

    #Helper classes
    def _chest_region_type_assert(self,chest_regions=None,chest_types=None,pairs=None):
        if chest_regions is not None:
            if len(chest_regions.shape) != 1:
                raise ValueError(\
                             'chest_regions must be a 1D array with elements in [0, 255]')
            if np.max(chest_regions) > 255 or np.min(chest_regions) < 0:
                raise ValueError(\
                             'chest_regions must be a 1D array with elements in [0, 255]')
        if chest_types is not None:
            if len(chest_types.shape) != 1:
                raise ValueError(\
                               'chest_types must be a 1D array with elements in [0, 255]')
            if np.max(chest_types) > 255 or np.min(chest_types) < 0:
                raise ValueError(\
                              'chest_types must be a 1D array with elements in [0, 255]')
        if pairs is not None:
            if len(pairs.shape) != 2:
                raise ValueError(\
                               'cpairs must be a 1D array with elements in [0, 255]')
            if np.max(chest_types) > 255 or np.min(chest_types) < 0:
                raise ValueError(\
                                'chest_types must be a 1D array with elements in [0, 255]')


class RegionTypeParser():
    """Parses the chest-region chest-type input data to identify all existing
    chest regions, chest types, and region-type pairs.

    Parameters
    ----------
    data : ND array
        The input data. Each value is assumed to be an unsigned short (16 bit)
        data type, where the least significant 8 bits encode the chest region,
        and the most significant 8 bits encode the chest type.

    Attributes
    ----------
    labels_ : array, shape ( M )
        The M unique labels in the data set
    """

    def __init__(self, data):        
        self._data = data
        assert len(data.shape) > 0, "Empty data set"

        self.labels_ = np.trim_zeros(np.unique(self._data))
        

        ##########
        # Changed in Aug 21 2019 to speed phenotyping. To avoid continuous
        # evaluation labelmap
        ##########
        self._data_indices = dict()
        for ll in set(self.labels_):
            self._data_indices[ll] = np.where(self._data==ll)

   
    def get_mask(self, chest_region=None):
        """Get's boolean mask of all data indices that match the chest-region
        chest-type query.

        If only a chest region is specified, then all voxels in that region
        will be included in the mask, regardless of the voxel's chest type
        value (chest region hierarchy is honored). If only a type is specified,
        then all voxels having that type will be included in the mask,
        regardless of the voxel's chest region. If both a region and type are
        speficied, then only those voxels matching both the region and type
        will be included in the mask (the chest region hierarchy is honored).

        Parameters
        ----------
        chest_region : int
            Integer value in the interval [0, 255] that indicates the chest
            region

        chest_type : int
            Integer value in the interval [0, 255] that indicates the chest
            type

        Returns
        -------
        mask : array, shape ( X, Y, Z )
            Boolean mask of all data indices that match the chest-region
            chest-type query. The chest region hierarchy is honored.
        """
        if chest_region is not None:
            if type(chest_region) != int and type(chest_region) != np.int64 \
              and type(chest_region) != np.int32:
                raise ValueError(
                    'chest_region must be an int between 0 and 255 inclusive')
      
        
        conventions = ChestConventions()
             
        mask_labels = []
        mask_labels.append(chest_region)
        #for l in self.labels_:
        #    r = class_map[l]

        #    if chest_region is not None:
            
        #        mask_labels.append(l)

        
                
        mask = np.empty(self._data.shape, dtype=bool)
        mask[:] = False


        ##########
        # Changed in Aug 21 2019 to speed phenotyping. To avoid continuous
        # evaluation labelmap
        ##########
        # PREVIUS CODE
        # for ll in set(mask_labels):
        #     tic = time.time()
        #     mask[self._data == ll] = True
        for ll in set(mask_labels):
            mask[self._data_indices[ll]] = True

        return mask

    def get_chest_regions(self):
        """Get the explicit list of chest regions in the data set.

        Returns
        -------
        chest_regions : array, shape ( N )
            Explicit list of the N chest regions in the data set
        """
        conventions = ChestConventions()

        tmp = []
        for l in self.labels_:            
            tmp.append(class_map[l])

        chest_regions = np.unique(np.array(tmp, dtype=int))
        return chest_regions

    def get_all_chest_regions(self):
        """Get all the chest regions in the data set, including those
        implicitly present as a result of the region hierarchy.

        Returns
        -------
        chest_regions : array, shape ( N )
            All chest regions in the data set
        """
        c = ChestConventions()
        num_regions = 105

        tmp = []
        for l in self.labels_:
            #r = class_map[l]
            tmp.append(l)


        chest_regions = np.unique(np.array(tmp, dtype=int))
        return chest_regions

    


class ParenchymaPhenotypes(Phenotypes):
    """General purpose class for generating parenchyma-based phenotypes.

    The user can specify chest regions, chest types, or region-type pairs over
    which to compute the phenotypes. Otherwise, the phenotypes will be computed
    over all structures in the specified labelmap. The following phenotypes are
    computed using the 'execute' method:
    'LAA950': fraction of the structure's region with CT HU values < -950
    'LAA925': fraction of the structure's region with CT HU values < -925
    'LAA910': fraction of the structure's region with CT HU values < -910
    'LAA905': fraction of the structure's region with CT HU values < -905
    'LAA900': fraction of the structure's region with CT HU values < -900
    'LAA875': fraction of the structure's region with CT HU values < -875
    'LAA856': fraction of the structure's region with CT HU values < -856
    'HAA700': fraction of the structure's region with CT HU values > -700
    'HAA600': fraction of the structure's region with CT HU values > -600
    'HAA500': fraction of the structure's region with CT HU values > -500
    'HAA250': fraction of the structure's region with CT HU values > -250
    'Perc10': HU value at the 10th percentile of the structure's HU histogram
    'Perc15': HU value at the 15th percentile of the structure's HU histogram
    'HUMean': Mean value of the structure's HU values
    'HUStd': Standard deviation of the structure's HU values
    'HUKurtosis': Kurtosis of the structure's HU values. Fisher's definition is
    used, meaning that normal distribution has kurtosis of 0. The calculation
    is corrected for statistical bias.
    'HUSkewness': Skewness of the structure's HU values. The calculation is
    corrected for statistical bias.
    'HUMode': Mode of the structure's HU values
    'HUMedian': Median of the structure's HU values
    'HUMin': Min HU value for the structure
    'HUMax': Max HU value for the structure
    'HUMean500': Mean CT value of the structure, but only considering CT
    values that are < -500 HU
    'HUStd500': Standard deviation of the structure's CT values, but only
    considering CT values that are < -500 HU
    'HUKurtosis500': Kurtosis of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUSkewness500': Skewness of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUMode500': Mode of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUMedian500': Median of the structure's HU values, but only
    considering CT values that are < -500 HU
    'HUMin500': Min HU value for the structure, but only considering CT values
    that are < -500 HU
    'HUMax500': Max HU value for the structure, but only considering CT values
    that are < -500 HU
    'HUMean950': Mean CT value of the structure, but only considering CT
    values that are < -950 HU
    'HUStd950': Standard deviation of the structure's CT values, but only
    considering CT values that are < -950 HU
    'HUKurtosis950': Kurtosis of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUSkewness950': Skewness of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUMode950': Mode of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUMedian950': Median of the structure's HU values, but only
    considering CT values that are < -950 HU
    'HUMin950': Min HU value for the structure, but only considering CT values
    that are < -950 HU
    'HUMax950': Max HU value for the structure, but only considering CT values
    that are < -950 HU
    'Volume': Volume of the structure, measured in liters
    'Mass': Mass of the structure measure in grams    
    'TypeFrac': The fraction of a type in a specified chest-region chest-type
    pair within the chest region of that pair.
    
    Parameters
    ----------
    chest_regions : list of strings
        List of chest regions over which to compute the phenotypes.

    chest_types : list of strings
        List of chest types over which to compute the phenotypes.

    pairs : list of lists of strings
        Each element of the list is a two element list indicating a region-type
        pair for which to compute the phenotypes. The first entry indicates the
        chest region of the pair, and second entry indicates the chest type of
        the pair.

    pheno_names : list of strings, optional
        Names of phenotypes to compute. These names must conform to the
        accepted phenotype names, listed above. If none are given, all
        will be computed.
    """
    def __init__(self, chest_regions=None,
                 pheno_names=None):
        
        c = ChestConventions()


        def get_key(val):
            for key, value in class_map.items():
                if val == value:    
                    return key
        self.chest_regions_ = None
        if chest_regions is not None:
            tmp = []
            for m in range(0, len(chest_regions)):
                tmp.append(get_key(chest_regions[m]))
            self.chest_regions_ = np.array(tmp)

                
        self.requested_pheno_names = pheno_names
        
        self.deprecated_phenos_ = ['NormalParenchyma', 'PanlobularEmphysema',
                                   'ParaseptalEmphysema', 'MildCentrilobularEmphysema',
                                   'ModerateCentrilobularEmphysema', 'SevereCentrilobularEmphysema',
                                   'MildParaseptalEmphysema']

        Phenotypes.__init__(self)    

    def declare_pheno_names(self):
        """Creates the names of the phenotypes to compute

        Returns
        -------
        names : list of strings
            Phenotype names
        """
        

        
        #Get phenotypes list from ChestConventions
        c=ChestConventions()
        names = c.ParenchymaPhenotypeNames
        return names

    def get_cid(self):
        """Get the case ID (CID)

        Returns
        -------
        cid : string
            The case ID (CID)
        """
        return self.cid_

    def execute(self, lm, cid, spacing, ct=None, chest_regions=None, pheno_names=None):
        """Compute the phenotypes for the specified structures for the
        specified threshold values.

        The following values are computed for the specified structures.
        'LAA950': fraction of the structure's region with CT HU values < -950
        'LAA925': fraction of the structure's region with CT HU values < -925
        'LAA910': fraction of the structure's region with CT HU values < -910
        'LAA905': fraction of the structure's region with CT HU values < -905
        'LAA900': fraction of the structure's region with CT HU values < -900
        'LAA875': fraction of the structure's region with CT HU values < -875
        'LAA856': fraction of the structure's region with CT HU values < -856
        'HAA700': fraction of the structure's region with CT HU values > -700
        'HAA600': fraction of the structure's region with CT HU values > -600
        'HAA500': fraction of the structure's region with CT HU values > -500
        'HAA250': fraction of the structure's region with CT HU values > -250
        'Perc10': HU value at the 10th percentile of the structure's HU
        histogram
        'Perc15': HU value at the 15th percentile of the structure's HU
        histogram
        'HUMean': Mean value of the structure's HU values
        'HUStd': Standard deviation of the structure's HU values
        'HUKurtosis': Kurtosis of the structure's HU values. Fisher's definition
        is used, meaning that normal distribution has kurtosis of 0. The
        calculation is corrected for statistical bias.
        'HUSkewness': Skewness of the structure's HU values. The calculation is
        corrected for statistical bias.
        'HUMode': Mode of the structure's HU values
        'HUMedian': Median of the structure's HU values
        'HUMin': Min HU value for the structure
        'HUMax': Max HU value for the structure
        'HUMean950': Mean CT value of the structure, but only considering CT
        values that are < -950 HU
        'HUStd950': Standard deviation of the structure's CT values, but only
        considering CT values that are < -950 HU
        'HUKurtosis950': Kurtosis of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUSkewness950': Skewness of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUMode950': Mode of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUMedian950': Median of the structure's HU values, but only
        considering CT values that are < -950 HU
        'HUMin950': Min HU value for the structure, but only considering CT values
        that are < -950 HU
        'HUMax950': Max HU value for the structure, but only considering CT values
        that are < -950 HU
        'HUMean500': Mean CT value of the structure, but only considering CT
        values that are < -500 HU
        'HUStd500': Standard deviation of the structure's CT values, but only
        considering CT values that are < -500 HU
        'HUKurtosis500': Kurtosis of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUSkewness500': Skewness of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUMode500': Mode of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUMedian500': Median of the structure's HU values, but only
        considering CT values that are < -500 HU
        'HUMin500': Min HU value for the structure, but only considering CT
        values that are < -500 HU
        'HUMax500': Max HU value for the structure, but only considering CT
        values that are < -500 HU
        'Volume': Volume of the structure, measured in liters
        'Mass': Mass of the structure measure in grams
        'TypeFrac': The fraction of a type in a specified chest-region 
        chest-type pair within the chest region of that pair.
        
        Parameters
        ----------
        lm : array, shape ( X, Y, Z )
            The 3D label map array

        cid : string
            Case ID

        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the CT volume

        ct : array, shape ( X, Y, Z )
            The 3D CT image array
                        
        chest_regions : array, shape ( R ), optional
            Array of integers, with each element in the interval [0, 255],
            indicating the chest regions over which to compute the LAA. If none
            specified, the chest regions specified in the class constructor
            will be used. If chest regions, chest types, and chest pairs are
            left unspecified both here and in the constructor, then the
            complete set of entities found in the label map will be used.

        chest_types : array, shape ( T ), optional
            Array of integers, with each element in the interval [0, 255],
            indicating the chest types over which to compute the LAA. If none
            specified, the chest types specified in the class constructor
            will be used. If chest regions, chest types, and chest pairs are
            left unspecified both here and in the constructor, then the
            complete set of entities found in the label map will be used.

        pairs : array, shape ( P, 2 ), optional
            Array of chest-region chest-type pairs over which to compute the
            LAA. The first column indicates the chest region, and the second
            column indicates the chest type. Each element should be in the
            interal [0, 255]. If none specified, the pairs specified in the
            class constructor will be used. If chest regions, chest types, and
            chest pairs are left unspecified both here and in the constructor,
            then the complete set of entities found in the label map will be
            used.

        pheno_names : list of strings, optional
            Names of phenotypes to compute. These names must conform to the
            accepted phenotype names, listed above. If none are given, all
            will be computed. Specified names given here will take precedence
            over any specified in the constructor.

        Returns
        -------
        df : pandas dataframe
            Dataframe containing info about machine, run time, and chest region
            chest type phenotype quantities.         
        """
        c = ChestConventions()

        if ct is not None:
            assert len(ct.shape) == len(lm.shape), \
              "CT and label map are not the same dimension"    

        dim = len(lm.shape)
        if ct is not None:
            for i in range(0, dim):
                assert ct.shape[0] == lm.shape[0], \
                  "Disagreement in CT and label map dimension"

        assert type(cid) == str, "cid must be a string"
        self.cid_ = cid
        self._spacing = spacing


        #Derive phenos to compute from list provided
        # As default use the list that is provided in the constructor of phenotypes
        phenos_to_compute = self.pheno_names_
        
        if pheno_names is not None:
            phenos_to_compute = pheno_names
        elif self.requested_pheno_names is not None:
            phenos_to_compute = self.requested_pheno_names

        #Deal with wildcard: defaul to main pheno list if wildcard is provided
        if c.GetChestWildCardName() in phenos_to_compute:
            phenos_to_compute = self.pheno_names_


        #Check validity of phenotypes
        for pheno_name in phenos_to_compute:
            assert pheno_name in self.pheno_names_, \
                  "Invalid phenotype name " + pheno_name

        #Warn for phenotpyes that have been deprecated
        for p in phenos_to_compute:
            if p in self.deprecated_phenos_:
                warnings.warn('{} is deprecated. Use TypeFrac instead.'.\
                              format(p), DeprecationWarning)
            
        rs = np.array([])
        
        if chest_regions is not None:
            rs =  chest_regions_
        elif self.chest_regions_ is not None:
            rs = self.chest_regions_

        parser = RegionTypeParser(lm)
        
        if rs.size == 0:
            rs = parser.get_all_chest_regions()

       
   


        # Now compute the phenotypes and populate the data frame
        for r in rs:
            if r != 0:
                mask_region = parser.get_mask(chest_region=r)
                self.add_pheno_group(ct, mask_region, mask_region, class_map[r],
                    c.GetChestWildCardName(), phenos_to_compute)


        return self._df

    def add_pheno_group(self, ct, mask, mask_region, chest_region, chest_type, phenos_to_compute):
        """This function computes phenotypes and adds them to the dataframe with
        the 'add_pheno' method. 
        
        Parameters
        ----------
        ct : array, shape ( X, Y, Z )
            The 3D CT image array

        mask : boolean array, shape ( X, Y, Z ), optional
            Boolean mask where True values indicate presence of the structure
            of interest
            
        mask_region: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for the corresponding region
            
        mask_type: boolean array, shape ( X, Y, Z ), optional
            Boolean mask for the corresponding type

        chest_region : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        chest_type : string
            Name of the chest region in the (region, type) key used to populate
            the dataframe

        phenos_to_compute : list of strings
            Names of the phenotype used to populate the dataframe

        References
        ----------
        1. Schneider et al, 'Correlation between CT numbers and tissue
        parameters needed for Monte Carlo simulations of clinical dose
        distributions'
        """
        mask_sum = np.sum(mask)
        print ("Computing phenos for region %s "%(chest_region))
        if ct is not None:
            ct_mask=ct[mask]
            hus=dict()

            ##########
            # Changed in Aug 21 2019 to speed phenotyping. To avoid continuous
            # evaluation labelmap
            ##########
            # PREVIUS CODE
            # hus[500] = ct[np.logical_and(mask, ct < -500)]
            # hus[950] = ct[np.logical_and(mask, ct < -950)]
            hus[500] = ct_mask[ct_mask < -500]
            hus[950] = ct_mask[ct_mask < -950]

            for pheno_name in phenos_to_compute:
                assert pheno_name in self.pheno_names_, \
                  "Invalid phenotype name " + pheno_name
                pheno_val = None
                if pheno_name == 'LAA950' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -950.)) / mask_sum
                elif pheno_name == 'LAA925' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -925.)) / mask_sum
                elif pheno_name == 'LAA910' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -910.)) / mask_sum
                elif pheno_name == 'LAA905' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -905.)) / mask_sum
                elif pheno_name == 'LAA900' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -900.)) / mask_sum
                elif pheno_name == 'LAA875' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -875.)) / mask_sum
                elif pheno_name == 'LAA856' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask < -856.)) / mask_sum
                elif pheno_name == 'HAA700' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -700.)) / mask_sum
                elif pheno_name == 'HAA600' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -600)) / mask_sum
                elif pheno_name == 'HAA500' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -500)) / mask_sum
                elif pheno_name == 'HAA250' and mask_sum > 0:
                    pheno_val = float(np.sum(ct_mask > -250)) / mask_sum
                elif pheno_name == 'Perc15' and mask_sum > 0:
                    pheno_val = np.percentile(ct_mask, 15)
                elif pheno_name == 'Perc10' and mask_sum > 0:
                    pheno_val = np.percentile(ct_mask, 10)
                elif pheno_name == 'HUMean' and mask_sum > 0:
                    pheno_val = np.mean(ct_mask)
                elif pheno_name == 'HUStd' and mask_sum > 0:
                    pheno_val = np.std(ct_mask)
                elif pheno_name == 'HUKurtosis' and mask_sum > 0:
                    pheno_val = kurtosis(ct_mask, bias=False, fisher=True)
                elif pheno_name == 'HUSkewness' and mask_sum > 0:
                    pheno_val = skew(ct_mask, bias=False)
                elif pheno_name == 'HUMode' and mask_sum > 0:
                    min_val = np.min(ct_mask.clip(-3000))
                    pheno_val = np.argmax(np.bincount(ct_mask.clip(-3000) + \
                        np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'HUMedian' and mask_sum > 0:
                    pheno_val = np.median(ct_mask)
                elif pheno_name == 'HUMin' and mask_sum > 0:
                    pheno_val = np.min(ct_mask)
                elif pheno_name == 'HUMax' and mask_sum > 0:
                    pheno_val = np.max(ct_mask)
                elif pheno_name == 'HUMean500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.mean(hus[500])
                elif pheno_name == 'HUStd500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.std(hus[500])
                elif pheno_name == 'HUKurtosis500' and mask_sum > 0:
                    if hus[500].shape[0]:
                        pheno_val = kurtosis(hus[500], bias=False, fisher=True)
                elif pheno_name == 'HUSkewness500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = skew(hus[500], bias=False)
                elif pheno_name == 'HUMode500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        min_val = np.min(hus[500].clip(-3000))
                        pheno_val = np.argmax(np.bincount(hus[500].clip(-3000) +\
                            np.abs(min_val))) - np.abs(min_val)                
                elif pheno_name == 'HUMedian500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.median(hus[500])
                elif pheno_name == 'HUMin500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.min(hus[500])
                elif pheno_name == 'HUMax500' and mask_sum > 0:
                    if hus[500].shape[0] > 0:
                        pheno_val = np.max(hus[500])
                elif pheno_name == 'HUMean950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.mean(hus[950])
                elif pheno_name == 'HUStd950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.std(hus[950])
                elif pheno_name == 'HUKurtosis950' and mask_sum > 0:
                    if hus[950].shape[0]:
                        pheno_val = kurtosis(hus[950], bias=False, fisher=True)
                elif pheno_name == 'HUSkewness950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = skew(hus[950], bias=False)
                elif pheno_name == 'HUMode950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        min_val = np.min(hus[950].clip(-3000))
                        pheno_val = np.argmax(np.bincount(hus[950].clip(-3000) +\
                            np.abs(min_val))) - np.abs(min_val)
                elif pheno_name == 'HUMedian950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.median(hus[950])
                elif pheno_name == 'HUMin950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.min(hus[950])
                elif pheno_name == 'HUMax950' and mask_sum > 0:
                    if hus[950].shape[0] > 0:
                        pheno_val = np.max(hus[950])
                elif pheno_name == 'Volume':
                    #Value in liters
                    pheno_val = np.prod(self._spacing)*float(mask_sum)/1e6
                elif pheno_name == 'Mass' and mask_sum > 0:
                    # This quantity is computed in a piecewise linear form
                    # according to the prescription presented in ref. [1].
                    # Mass is computed in grams. First compute the
                    # contribution in HU interval from -98 and below.
                    pheno_val = 0.0
                    HU_tmp = ct[np.logical_and(mask, ct < -98)].clip(-1000)
                    if HU_tmp.shape[0] > 0:
                        m = (1.21e-3-0.93)/(-1000+98)
                        b = 1.21e-3 + 1000*m
                        pheno_val += np.sum((m*HU_tmp + b)*\
                            np.prod(self._spacing)*0.001)
        
                    # Now compute the mass contribution in the interval
                    # [-98, 18] HU. Note the in the original paper, the
                    # interval is defined from -98HU to 14HU, but we
                    # extend in slightly here so there are no gaps in
                    # coverage. The values we report in the interval
                    # [14, 23] should be viewed as approximate.
                    HU_tmp = \
                      ct[np.logical_and(np.logical_and(mask, ct >= -98),
                            ct <= 18)]
                    if HU_tmp.shape[0] > 0:
                        pheno_val += \
                          np.sum((1.018 + 0.893*HU_tmp/1000.0)*\
                            np.prod(self._spacing)*0.001)
        
                    # Compute the mass contribution in the interval
                    # (18, 100]
                    HU_tmp = \
                      ct[np.logical_and(np.logical_and(mask, ct > 18),
                                               ct <= 100)]
                    if HU_tmp.shape[0] > 0:
                        pheno_val += np.sum((1.003 + 1.169*HU_tmp/1000.0)*\
                            np.prod(self._spacing)*0.001)
        
                    # Compute the mass contribution in the interval > 100
                    HU_tmp = ct[np.logical_and(mask, ct > 100)]
                    if HU_tmp.shape[0] > 0:
                        pheno_val += np.sum((1.017 + 0.592*HU_tmp/1000.0)*\
                            np.prod(self._spacing)*0.001)
                if pheno_val is not None:
                    self.add_pheno([chest_region, chest_type],
                                   pheno_name, pheno_val)
                    
        if 'TypeFrac' in phenos_to_compute and mask is not None and \
            mask_region is not None:
            denom = np.sum(mask_region)
            if denom > 0:
                pheno_val = float(np.sum(mask))/float(denom)
                self.add_pheno([chest_region, chest_type],
                                'TypeFrac', pheno_val)

if __name__ == "__main__":
    desc = """Generates parenchyma phenotypes given input CT and segmentation \
    data. Note that in general, specified chest-regions, chest-types, and \
    region-type pairs indicate the parts of the image over which the requested \
    phenotypes are computed. An exception is the TypeFrac phenotype; in that 
    case, a region-type pair must be specified. The TypeFrac phenotype is \
    computed by calculating the fraction of the pair's chest-region that is \
    covered by the pair's chest-type. For example, if a user is interested in \
    the fraction of paraseptal emphysema in the left lung, he/she would \
    specify the TypeFrac phenotype and the LeftLung,ParaseptalEmphysema \
    region-type pair."""

    parser = ArgumentParser(description=desc)
    
    parser.add_argument('--in_ct', '-in_ct', required=True,
                      help='Input CT file')

    parser.add_argument('--in_lm', '-in_lm', required=True,
                      help='Input label map containing structures of interest')

    parser.add_argument('--out_csv', '-out_csv', required=True,
                      help='Output csv file in which to store the computed dataframe')

    parser.add_argument('--cid', '-cid', required=True, help='Case id')

    # Keep these arguments as strings for compatibility purposes
    parser.add_argument('-r', dest='chest_regions',
                      help='Chest regions. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired regions over which to compute the phenotypes. \
                      E.g. LeftLung,RightLung would be a valid input.')



    args = parser.parse_args()

    image_io = ImageReaderWriter()

    regions = None
    if args.chest_regions is not None:
        regions = args.chest_regions.split(',')
    
    lm, lm_header = image_io.read_in_numpy(args.in_lm)

    ct, ct_header = image_io.read_in_numpy(args.in_ct)

    spacing = lm_header['spacing']


    paren_pheno = ParenchymaPhenotypes(chest_regions=regions)

    df = paren_pheno.execute(lm, args.cid, spacing, ct=ct)

    df.to_csv(args.out_csv, index=False)







