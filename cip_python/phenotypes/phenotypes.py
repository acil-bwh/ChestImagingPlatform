import platform
import datetime
import numpy as np
import pandas as pd
from cip_python.common import ChestConventions

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
        c = ChestConventions()
        valid_values = dict()
    
        region_names = [c.GetChestWildCardName()]
        for i in range(0, c.GetNumberOfEnumeratedChestRegions()):
            region_names.append(c.GetChestRegionName(i))
        valid_values['Region'] = region_names

        type_names = [c.GetChestWildCardName()]
        for i in range(0, c.GetNumberOfEnumeratedChestTypes()):
            type_names.append(c.GetChestTypeName(i))
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
        c = ChestConventions()

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
                tmp[self.key_names_[i]] = [key_value[i]]
            self._df = pd.concat([self._df, pd.DataFrame(tmp)], ignore_index=True,sort=False)
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


