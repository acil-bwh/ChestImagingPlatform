import numpy as np
import pdb
import copy
import scipy.ndimage.morphology as scimorph
from cip_python.segmentation.chest_partition import ChestPartition


class AnteriorPosteriorPartition(ChestPartition):
    """General class for partition generation in the AP axis.
    """
    def __init__(self): #, rind_width = None, rind_ratio = None):
        """
        """

        ChestPartition.__init__(self) 
        
    def declare_partition_regions(self):
        partition_regions = ['AP1of4','AP2of4','AP3of4','AP4of4']
        
        return partition_regions
       
    def execute(self, lung_labelmap, spacing, chest_partitions):
        ##The used requests partitions, return a a dictionary of binary labelmaps
        ## No partition: WildCard.
        """
        compute the labelmap containing the partition regions
        
        Parameters
        lm : array, shape ( X, Y, Z )
            The 3D lung label map array
            
        spacing : array, shape ( 3 )
            The x, y, and z spacing, respectively, of the labelmap    
        
        
        chest_partitions: If
            left unspecified , then the
            complete set of possible partitions will be used.

        Returns
        -------
        partition_labelmaps: dict()  
            Doctionary of labelmaps for all the requested partitions.
        """
                
        """ Compute the distance map associated with lung_labelmap """
        binary_labelmap = copy.copy(lung_labelmap)
        binary_labelmap[binary_labelmap > 0] = 1
        
        """ find labelmap bounadries """
        
        xmax, ymax,zmax = np.max(np.where(binary_labelmap>0), 1)
        xmin, ymin,zmin = np.min(np.where(binary_labelmap>0), 1)       
        
        """ find number of location along AP (Y) axis """
        slice_numbers = np.linspace( ymin,ymax,4, endpoint=False)[1:,]
        
        print(slice_numbers)
        partition_labelmaps=dict()
        """ return 1 labelmap per requested partition. These should be boolean"""   
        for partition_name in chest_partitions:
            if partition_name == self.get_partition_wildcard_name():
                partition_labelmaps['WildCard']=np.zeros(np.shape(lung_labelmap), dtype=bool)
                partition_labelmaps['WildCard'][binary_labelmap > 0] = 1  
 
            else:
                assert partition_name in self.partition_regions_, \
                  "Invalid partition name " + partition_name

                if partition_name == 'AP1of4':                                    
                    partition_labelmaps['AP1of4']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['AP1of4'][:,0:int(slice_numbers[0]),:]=1 
                    partition_labelmaps['AP1of4'][lung_labelmap==0]=0 
                elif partition_name == 'AP2of4':
                    partition_labelmaps['AP2of4']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['AP2of4'][:,int(slice_numbers[0]):int(slice_numbers[1]),:]=1 
                    partition_labelmaps['AP2of4'][lung_labelmap==0]=0               
                elif partition_name == 'AP3of4':
                    partition_labelmaps['AP3of4']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['AP3of4'][:,int(slice_numbers[1]):int(slice_numbers[2]),:]=1 
                    partition_labelmaps['AP3of4'][lung_labelmap==0]=0  
                elif partition_name == 'AP4of4':
                    partition_labelmaps['AP4of4']= np.zeros(np.shape(lung_labelmap), dtype=bool)
                    partition_labelmaps['AP4of4'][:,int(slice_numbers[2]):,:]=1  
                    partition_labelmaps['AP4of4'][lung_labelmap==0]=0  
   
            
        return partition_labelmaps
        
