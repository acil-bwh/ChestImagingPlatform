import numpy as np

#class ChestPartitionRegionConventions:
#    """ Class defining chest partition conventions. For now a simple class where a list
#        defines the relationship between the partition_region name and the partition_region 
#        value in a labelmap.
#    
#    Attributes
#    ----------
#    partition_region_list_ : list of tuple
#            Tuples comsisting name of regions and their associated labelmap value 
#
#    val_to_name_ : dict
#            dictionary that maps from partition region label value to partition region
#                name
#
#    val_to_name_ : dict
#            dictionary that maps from partition region name to partition region
#                label value 
#    """   
#    
#    def __init__(self):
#        self.val_to_name_ = {} #AtoB
#        self.name_to_val_ = {}
#        
#        partition_region_list_ = [['WholeLung', 1],['UpperThird', 2], ['MiddleThird', 3],\
#            ['LowerThird', 4], ['Rind', 5], ['Core', 6], ['Rind10', 7], ['Core10', 8],\
#            ['Rind15', 9], ['Core15', 10], ['Rind20', 11], ['Core20', 12]]
#
#        for partition_region in partition_region_list_:
#            self.add_partition_region(partition_region[1], partition_region[0])
#
#    def add_partition_region(self, val, name):
#
#        """
#        Adds a partition region to the conventions
#        
#        Parameters
#        val : int
#            Labelmap value of the partitoin region
#            
#        name : string
#            Name of the partitoin region   
#        
#        Returns
#        -------
#        None  
#        """        
#        self.val_to_name_[val] = name
#        self.name_to_val_[name] = val
#
#    def get_partition_region_name_from_value(self, val):
#        """
#        get a partition region name from labelmap value
#        
#        Parameters
#        -------
#        val : int
#            Labelmap value of the partitoin region
#            
#        
#        Returns
#        -------
#        name : string
#            Name of the partition region     
#        """   
#        if val in self.val_to_name_:
#            return self.val_to_name_[val]
#        return None
#
#    def get_partition_region_value_from_name(self, name):
#        """
#        get a partition region name from labelmap value
#        
#        Parameters
#        -------                
#        name : string
#            Name of the partition region   
#                    
#        Returns
#        -------       
#        val : int
#            Labelmap value of the partitoin region
#              
#        """   
#        if name in self.name_to_val_:
#            return self.name_to_val_[name]
#        return None


        
class ChestPartition:
    """Base class for partition genearting classes.
    
    Attributes
    ----------
    partition_regions_ : list of strings
                          Name of the partitions

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
        
        #cp = ChestPartitionRegionConventions()
        self.partition_regions_ = self.declare_partition_regions()
  
        #self.valid_key_values_ = []
        #for partition_region in self.partition_regions_: 
        #    self.valid_key_values_.append(\
        #        cp.get_partition_region_value_from_name(partition_region)) 
                        
    def get_partition_wildcard_name(self):
        #part_regions.extend(['WildCard'])
        return 'WildCard'
        
    def execute(self):
        pass
        
    
        
#    def CheckSubordinateSuperiorChestPatitionRelationship(subordinate, superior):
#        subordinateTemp = subordinate;
#
#        while (s_ChestConventions.ChestRegionHierarchyMap.find(subordinateTemp) !=
#            s_ChestConventions.ChestRegionHierarchyMap.end()): 
#                if (s_ChestConventions.ChestRegionHierarchyMap[subordinateTemp] == superior): 
#                    return True
#            
#                else: 
#                    subordinateTemp = s_ChestConventions.ChestRegionHierarchyMap[subordinateTemp]
#
#        return False
        
#    def get_partition_region_mask(self, partition_region): # Can potentially have multiple partition regions
#        # and use and 'and operator'    
#        """ Get a mask corresponding to the partition defined by the partition region
#        
#        Parameters
#        ----------
#        partition_region : string
#            The name of the partition region for which we need to obtain the mask
#        
#        Returns
#        -------
#        partition_mask : array, shape ( X, Y, Z )
#            The 3D mask array for the requested partition          
#        """
#        
#        assert partition_region in self.partition_regions_        
#
#        cp = ChestPartitionRegionConventions()
#
#        partition_val = cp.get_partition_region_value_from_name(partition_region)       
#        partition_mask = np.zeros_like(self.partition_labelmap_)                
#        
#        # First get all label values in the labelmap that are children 
#        # of partition_region
#        
#        # then for each child, assign all voxels with that labels to 1
#        
#        partition_mask[self.partition_labelmap_ == partition_val] = 1
#
#        return partition_mask
        
    #def get_all_partition_region_values(self): 
    #    """ all the partition_region label values for the current partition
    #    
    #    Parameters
    #    ----------
    #    None
    #    
    #    Returns
    #    -------
    #     : list of ints
    #    All partition region label values         
    #    """
    #    
    #    return np.unique(self.partition_labelmap_)
        
    def get_all_partition_region_names(self):
        """ all the partition_region names for the current partition
        
        Parameters
        ----------
        None
        
        Returns
        -------
        unique_names : list of strings
        All partition region label names         
        """
        
        return self.partition_regions_
        
        
        #cp = ChestPartitionRegionConventions()
        #unique_values = np.unique(self.partition_labelmap_)
        #unique_names = []
        #for unique_value in unique_values: 
        #    assert unique_value in self.valid_key_values_
        #    unique_names = unique_names.append(\
        #        cp.get_partition_region_name_from_value(unique_value)) 
        

        
