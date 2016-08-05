import numpy as np
from ..common import ChestConventions
     
class DistExtractorFromROI:
    """General purpose class implementing a distance feature
    extractor from the CT patch. 

    The user inputs CT patch and a labelmap mask. The output is the average 
    distance to . 
       
    Parameters 
    ----------    
    chest_region : string
        Chest region over which the distance was computed. This will be 
        added to the dataframe as a column.
        
    chest_type : string
        Chest type over which the distance was computed.

    pairs : lists of strings
        Two element list indicating a region-type pair for which the distance
        was computed. The first entry indicates the chest region of the pair, 
        and second entry indicates the chest type of the pair. If more than 1 of
        chest_region, chest_type, pairs is specified Region will superceed type,
        and type will superceed pairs.
 
 
                             
    Attribues
    ---------
    distance_feature_name : String
        Name of the region and type from which distance was computed
        
    dist_ : float
        Average distance value of the patch.       
    """
    def __init__(self, chest_region=None, chest_type=None, pair=None):

        c = ChestConventions()
                
        if chest_region is not None:
             distance_region_type = c.GetChestRegionName(\
                c.GetChestRegionValueFromName(chest_region))
        elif chest_type is not None:
             distance_region_type = c.GetChestTypeName(\
                c.GetChestTypeValueFromName(chest_type))
        elif pair is not None:
            assert len(pair)%2 == 0, "Specified region-type pair not understood"   
            r = c.GetChestRegionName(c.GetChestRegionValueFromName(pair[0]))
            t = c.GetChestTypeName(c.GetChestTypeValueFromName(pair[1]))
            distance_region_type = r+t
            
        self.distance_feature_name = distance_region_type+"Distance"
        self.dist_ = None
            
       
                                                            

                
    def fit(self, ct_patch, lm_patch):
        """Compute the histogram of each patch defined in 'patch_labels' beneath
        non-zero voxels in 'lm' using the CT data in 'ct'.
        
        Parameters
        ----------
        
        ct_patch: 3D numpy array, shape (L, M, N)
            Input CT patch from which histogram information will be derived
    
        lm_patch: 3D numpy array, shape (L, M, N)
            Input mask mask histograms will be computed.    
        """                
            
        # extract the lung area from the CT for the patch
        patch_distances = ct_patch[(lm_patch >0)] 
        # linearize features
        if (np.shape(patch_distances)[0] > 1):
            distance_vector = np.array(patch_distances.ravel()).T

            self.dist_ = np.mean(distance_vector)
        


