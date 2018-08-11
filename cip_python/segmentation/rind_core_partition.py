import numpy as np
import pdb
import copy
import scipy.ndimage.morphology as scimorph
from cip_python.segmentation.chest_partition import ChestPartition


class RindCorePartition(ChestPartition):
    """General class for rind versus core partition generation.
    
    """
    def __init__(self): #, rind_width = None, rind_ratio = None):
        """
        """

        ChestPartition.__init__(self) 
        
    def declare_partition_regions(self):
        partition_regions = ['Rind10','Core10','Rind15','Core15','Rind20','Core20']
        
        return partition_regions
       
    def execute(self, lung_labelmap, spacing, chest_partitions=None):

        """
        compute the labelmap containing the partition regions
        
        Parameters
        lm : array, shape ( X, Y, Z )
            The 3D lung label map array
            
        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the labelmap    
        
        
        chest_partitions: If
            left unspecified both here and in the constructor, then the
            complete set of entities found in the label map will be used.

        Returns
        -------
        partition_labelmaps: dict()  
            Dictionary of labelmaps for all the requested partitions.
        """
        
        
        """ Compute the distance map associated with lung_labelmap """
        binary_labelmap = copy.copy(lung_labelmap)
        binary_labelmap[binary_labelmap > 0] = 1 
        
        # distance_transform_edt returns 0 whele labelmap is 0 and +ve
        # distance everywhere else 
        lung_distance_map = \
            scimorph.distance_transform_edt(binary_labelmap)
        
        if chest_partitions is None:
            chest_partitions = self.get_all_partition_region_names()
            #chest_partitions = self.partition_regions_
        
        
        partition_labelmaps=dict()
        """ return 1 labelmap per requested partition. These should be boolean"""   
        for partition_name in chest_partitions:
            if partition_name == self.get_partition_wildcard_name():
                partition_labelmaps['WildCard']=np.zeros(np.shape(lung_labelmap), dtype=bool)
                partition_labelmaps['WildCard'][binary_labelmap > 0] = 1  
 
            else:
                assert partition_name in self.partition_regions_, \
                  "Invalid partition name " + partition_name

                if partition_name == 'Rind10':
                    rind_width=10.0  
                    rind_width_voxels = rind_width/spacing[0]                                      
                    partition_labelmaps['Rind10']= np.zeros(np.shape(lung_labelmap), dtype=bool)#np.zeros_like(lung_labelmap)
                    partition_labelmaps['Rind10'][(lung_distance_map<=rind_width_voxels)]=1 
                    partition_labelmaps['Rind10'][lung_labelmap==0]=0 

                elif partition_name == 'Core10':
                    rind_width=10.0  
                    rind_width_voxels = rind_width/spacing[0]                                         
                    partition_labelmaps['Core10']= np.zeros(np.shape(lung_labelmap), dtype=bool)#np.zeros_like(lung_labelmap)
                    partition_labelmaps['Core10'][(lung_distance_map>rind_width_voxels)]=1 
                    partition_labelmaps['Core10'][lung_labelmap==0]=0                 

                elif partition_name == 'Rind15':
                    rind_width=15.0  
                    rind_width_voxels = rind_width/spacing[0]                                      
                    partition_labelmaps['Rind15']= np.zeros(np.shape(lung_labelmap), dtype=bool)#np.zeros_like(lung_labelmap)
                    partition_labelmaps['Rind15'][(lung_distance_map<=rind_width_voxels)]=1 
                    partition_labelmaps['Rind15'][lung_labelmap==0]=0 
    
                elif partition_name == 'Core15':
                    rind_width=15.0  
                    rind_width_voxels = rind_width/spacing[0]                                         
                    partition_labelmaps['Core15']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['Core15'][(lung_distance_map>rind_width_voxels)]=1 
                    partition_labelmaps['Core15'][lung_labelmap==0]=0            
            
                elif partition_name == 'Rind20':
                    rind_width=20.0  
                    rind_width_voxels = rind_width/spacing[0]                                      
                    partition_labelmaps['Rind20']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['Rind20'][(lung_distance_map<=rind_width_voxels)]=1 
                    partition_labelmaps['Rind20'][lung_labelmap==0]=0 
    
                elif partition_name == 'Core20':
                    rind_width=20.0  
                    rind_width_voxels = rind_width/spacing[0]                                         
                    partition_labelmaps['Core20']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['Core20'][(lung_distance_map>rind_width_voxels)]=1 
                    partition_labelmaps['Core20'][lung_labelmap==0]=0          
            
        return partition_labelmaps
                
        
