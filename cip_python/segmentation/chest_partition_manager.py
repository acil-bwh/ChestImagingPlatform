import numpy as np
import pdb
import scipy.ndimage.morphology as scimorph

class ChestPartitionRegionConventions:
    """ Class defining chest partition conventions. For now a simple class where a list
        defines the relationship between the partition_region name and the partition_region 
        value in a labelmap.
    
    Attributes
    ----------
    partition_region_list_ : list of tuple
            Tuples comsisting name of regions and their associated labelmap value 

    val_to_name_ : dict
            dictionary that maps from partition region label value to partition region
                name

    val_to_name_ : dict
            dictionary that maps from partition region name to partition region
                label value 
    """   
    
    def __init__(self):
        self.val_to_name_ = {} #AtoB
        self.name_to_val_ = {}
        
        partition_region_mapping_ = [['UndefinedPartition', 0],['WholeLung', 1],['UpperThird', 2], ['MiddleThird', 3],\
            ['LowerThird', 4], ['Rind', 5], ['Core', 6] ]

        self.partition_region_list_ = []
        for partition_region in partition_region_mapping_:
            self.add_partition_region(partition_region[1], partition_region[0])
            self.partition_region_list_.append(partition_region[0])

    def get_partition_region_list(self):
        return self.partition_region_list_

    def add_partition_region(self, val, name):

        """
        Adds a partition region to the conventions
        
        Parameters
        val : int
            Labelmap value of the partitoin region
            
        name : string
            Name of the partitoin region   
        
        Returns
        -------
        None  
        """        
        self.val_to_name_[val] = name
        self.name_to_val_[name] = val

    def get_partition_region_name_from_value(self, val):
        """
        get a partition region name from labelmap value
        
        Parameters
        -------
        val : int
            Labelmap value of the partitoin region
            
        
        Returns
        -------
        name : string
            Name of the partition region     
        """   
        if val in self.val_to_name_:
            return self.val_to_name_[val]
        return None

    def get_partition_region_value_from_name(self, name):
        """
        get a partition region name from labelmap value
        
        Parameters
        -------                
        name : string
            Name of the partition region   
                    
        Returns
        -------       
        val : int
            Labelmap value of the partitoin region
              
        """   
        if name in self.name_to_val_:
            return self.name_to_val_[name]
        return None


    def CheckSubordinateSuperiorChestRegionRelationship(self, subordinate, superior):
        """
        This method checks if the chest partition 'subordinate' is within
        the chest partition 'superior'. As it is, it assumes that all chest partitions are
        within the WHOLELUNG lung partition. This really is the only hierarchy
        that exists for partition regions since, within a partition, partition
        regions are mutually exclusive.
        
        Parameters
        -------                
        subordinate : string
            Name of the subordinate partition region   
                    
        superior : string
            Name of the superior partition region   
            
        Returns
        -------       
        is_subordinate_superior : boolean
            Equals true if the subordinate is within superior
        """
        if ((subordinate == 'UndefinedPartition') or (superior == 'UndefinedPartition')):
            return False
        if (superior == 'WholeLung'):
            return True
        if (subordinate == superior):
            return True
        
        return False
 
class ChestPartitionManager(object):

  def __init__(self, labelmap):
      self.labelmap = labelmap
      self.__distance_map__ = None
      
  @property
  def distance_map(self):
    if self.__distance_map__ is None:
        self.__distance_map__ = self._compute_distance_map_()
    return self.__distance_map__
                  
  def _compute_distance_map_(self):
      """Compute the distance map from self.volume"""
      self.labelmap[self.labelmap > 0] = 1
      # distance_transform_edt returns 0 whele labelmap is 0 and +ve
      # distance everywhere else 
      distance_map = \
            scimorph.distance_transform_edt(self.labelmap)
      return distance_map
      
  def get_partition_region_mask(self, partitionId, **kwargs):
      
    cp = ChestPartitionRegionConventions()  

    if partitionId == cp.get_partition_region_value_from_name('Core'): 
        assert (('rind_width' in kwargs) or  ('rind_ratio' in kwargs)), \
            "rind_width or rind_ratio must be set"
        assert ('spacing' in kwargs), "spacing must be set"
                    
        if ('rind_width' in kwargs):
            return self._get_core_(kwargs['spacing'], rind_width = kwargs['rind_width'])
        elif ('rind_ratio' in kwargs):
            return self._get_core_(kwargs['spacing'], rind_ratio = kwargs['rind_ratio'])

    if partitionId == cp.get_partition_region_value_from_name('Rind'): 
        assert (('rind_width' in kwargs) or  ('rind_ratio' in kwargs)), \
            "rind_width or rind_ratio must be set"
        assert ('spacing' in kwargs), "spacing must be set"
                    
        if ('rind_width' in kwargs):
            return self._get_rind_(kwargs['spacing'], rind_width = kwargs['rind_width'])
        elif ('rind_ratio' in kwargs):
            return self._get_rind_(kwargs['spacing'], rind_ratio = kwargs['rind_ratio'])
            

    if partitionId == cp.get_partition_region_value_from_name('WholeLung'): 
        return self._get_whole_lung_()
         
  
  def _get_whole_lung_(self, spacing, rind_width = None, rind_ratio = None):
    partition_labelmap = np.ones_like(self.labelmap)
    partition_labelmap[self.labelmap==0]=0        
    return partition_labelmap
                      
  def _get_core_(self, spacing, rind_width = None, rind_ratio = None):
    if rind_width is  None:
        rind_width = self.distance_map.max()*rind_ratio
        
    rind_width_voxels = rind_width/spacing[0]
    partition_labelmap = np.zeros_like(self.labelmap)
        
    partition_labelmap[(self.distance_map>rind_width_voxels) ] = 1    
    partition_labelmap[self.labelmap==0]=0        
    return partition_labelmap

  def _get_rind_(self, spacing, rind_width = None, rind_ratio = None):
    if rind_width is  None:
        rind_width = self.distance_map.max()*rind_ratio
    rind_width_voxels = rind_width/spacing[0]
    partition_labelmap = np.zeros_like(self.labelmap)
    
    partition_labelmap[(self.distance_map<=rind_width_voxels)] = 1       
    partition_labelmap[self.labelmap==0]=0        
         
    return partition_labelmap

