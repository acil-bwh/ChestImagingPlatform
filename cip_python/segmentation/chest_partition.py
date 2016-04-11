import numpy as np


class ChestPartition:
    """Base class for partition genearting classes.
    
    Attributes
    ----------
    partition_regions_ : list of strings
                          Name of the regions that come with a partition  

    Notes
    -----
    This class is meant to be inherited from in order to obtain a specific
    chest partition.
    
    1) Overload the declare_partition_regions that will declare all the possible
         regions that can exist within a partinion
    2) Overload the 'execute' method that will compute the partition 
    3) Overload the 'get_partition_region_mask' method which returns a mask for a 
         given partition region.   


    """
    def __init__(self):
        """
        """
        self.partition_regions_ = self.declare_partition_regions()
        #self.valid_key_values_ = self.valid_key_values()
        self.partition_labelmap_ = None
        
        self.partition_region_conventions = {'WholeLung': 1, 'UpperThird': 2, 'MiddleThird': 3,\
           'LowerThird': 4, 'Rind' :  5, 'Core' : 6}
        
    def declare_partition_regions(self):
        pass
       
    def execute(self):
        pass
        
    def get_partition_region_mask(self, partition_region): # Can potentially have multiple partition regions
        # and use and 'and operator'    
        """ Get a mask corresponding to the partition defined by the partition region
        
        Parameters
        ----------
        partition_region : string
            The name of the partition region for which we need to obtain the mask
        
    
        """
        
        assert partition_region in self.partition_regions_        

        partition_val = self.partition_region_conventions[partition_region]        
        partition_mask = np.zeros_like(self.partition_labelmap_)                
        partition_mask[self.partition_labelmap_ == partition_val] = 1

        return partition_mask
        
