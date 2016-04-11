import numpy as np
import scipy.ndimage.morphology as scimorph
from cip_python.segmentation.chest_partition import ChestPartition
import pdb

class RindVsCorePartition(ChestPartition):
    """General class for rind versus core partition generation.
    
    Attributes
    ----------
    partition_regions_ : list of strings
                          Name of the regions that come with a partition     

    rind_windth_ : float
                    Width of the rind in mm
    rind_ratio_ : float
                  ratio between the rind width and the maximum width possible  
                  Either rind_windth_ or rind_ratio_ must be set. Not both.  
    """
    def __init__(self, rind_width = None, rind_ratio = None):
        """
        """
        
        assert ((rind_width is None) or (rind_ratio is None)), \
            "rind_windth and rind_ratio cannot both be set " 

        assert ((rind_width is not None) or (rind_ratio is not None)), \
            "Either rind_windth and rind_ratio must be set " 
            
        self.rind_width_ = rind_width
        self.rind_ratio_ = rind_ratio
        ChestPartition.__init__(self) 
        
    def declare_partition_regions(self):
        partition_regions = ['Rind', 'Core']
        
        return partition_regions
       
    def execute(self, lung_labelmap, spacing):
        
        """
        compute the labelmap containing the partition regions
        
        Parameters
        lm : array, shape ( X, Y, Z )
            The 3D lung label map array
            
        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the labelmap    
        
        Returns
        -------
        None  
        """
        
        
        # Compute the distance map associated with lung_labelmap
        lung_labelmap[lung_labelmap > 0] = 1
        # distance_transform_edt returns 0 whele labelmap is 0 and +ve
        # distance everywhere else 
        lung_distance_map = \
            scimorph.distance_transform_edt(lung_labelmap)
            
        if self.rind_width_ is not None:
            rind_width = self.rind_width_
        else:
            rind_width = lung_distance_map.max()*self.rind_ratio_
        
        # Convert the rind width from mm to voxels
        rind_width_voxels = rind_width/spacing[0]
        
        self.partition_labelmap_ = np.zeros_like(lung_labelmap)
        
        self.partition_labelmap_[(lung_distance_map<=rind_width_voxels)] = self.partition_region_conventions['Rind']
        self.partition_labelmap_[(lung_distance_map>rind_width_voxels) ] = self.partition_region_conventions['Core']
        
        self.partition_labelmap_[lung_labelmap==0]=0
        
        pdb.set_trace()