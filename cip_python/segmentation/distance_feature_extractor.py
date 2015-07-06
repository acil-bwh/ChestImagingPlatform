import scipy.io as sio
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import warnings
from sklearn.neighbors import KernelDensity
import nrrd
from kde_bandwidth import botev_bandwidth
from cip_python.ChestConventions import ChestConventions
#from cip_python.io.image_reader_writer import ImageReaderWriter
     
class DistanceFeatureExtractor:
    """General purpose class implementing a distance feature extractor. 

    The user inputs distance numpy array, a mask, and a patch segmentation. The
    output is a pandas dataframe with patch #=number and
    a associated distance entries
       
    Parameters 
    ----------
    input_datafrane: Pandas dataframe
        Contains feature information previously computed over the patches
        for which we seak the distance information    
        
    chest_region : string
        Chest region over which the distance was computed. This will be 
        added to the daraframe as a column.
        
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
    df_ : Pandas dataframe
        Contains the patch feature information. The 'patch_label' column
        corresponds to the segmented patch over which the distance is 
        computed. The 'chestwall_distance' column contains the physical distance 
        from the center of the patch to the chest wall.        
    """
    def __init__(self, chest_region=None, chest_type=None, pair=None, \
        input_dataframe=None):
        # get the region / type over which distance is computed
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
        assert  distance_region_type is not None, "region type not specified" 
           
        self.distance_feature_name = distance_region_type+"Distance"                                           
        # Initialize the dataframe
        if  input_dataframe is None:
            cols = ['patch_label', self.distance_feature_name]
            self.df_ = pd.DataFrame(columns=cols)
            print(cols)
        else:         
            self.df_ = input_dataframe.append(pd.DataFrame(columns=[self.distance_feature_name]))    
               
        
    def fit(self, distance_image, lm, patch_labels):
        """Compute the histogram of each patch defined in 'patch_labels' beneath
        non-zero voxels in 'lm' using the CT data in 'ct'.
        
        Parameters
        ----------
        patch_labels: 3D numpy array, shape (L, M, N)
            Input patch segmentation
        
        distance_image: 3D numpy array, shape (L, M, N)
            Input distance image 
    
        lm: 3D numpy array, shape (L, M, N)
            Input mask where distance features wil be extracted.    
        """        
        unique_patch_labels = np.unique(patch_labels)
        # loop through each patch 
        for p_label in unique_patch_labels:               
            # extract the lung area from the CT for the patch
            patch_distances = \
                distance_image[np.logical_and(patch_labels==p_label, lm >0)] 
        
            # linearize features
            distance_vector = np.array(patch_distances.ravel()).T
            if (np.shape(distance_vector)[0] > 1):
                # compute the average distance
                mean_dist = np.mean(distance_vector) 
                tmp = dict()
                tmp['patch_label'] = p_label
                tmp[self.distance_feature_name] = mean_dist

                # save in data frame 
                self.df_ = self.df_.append(tmp, ignore_index=True)
        
if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_distance',
                      help='Input distance map file', dest='in_distance', \
                      metavar='<string>', default=None)
    parser.add_option('--in_lm',
                      help='Input mask file. The histogram will only be \
                      computed in areas where the mask is > 0. If lm is not \
                      included, the histogram will be computed everywhere.', 
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--in_patches',
                      help='Input patch labels file. A label is defined for \
                      each corresponding CT voxel. A histogram will be \
                      computed for each patch label', dest='in_patches', 
                      metavar='<string>',default=None)
    parser.add_option('-r',
                      help='Chest regions. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired regions over which distance was computed. \
                      E.g. LeftLung,RightLung would be a valid input.',
                      dest='chest_region', metavar='<string>', default=None)
    parser.add_option('-t',
                      help='Chest types. Should be specified as a \
                      common-separated list of string values indicating the \
                      desired types over which distance was computed. \
                      E.g.: Vessel,NormalParenchyma would be a valid input.',
                      dest='chest_type', metavar='<string>', default=None)
    parser.add_option('-p',
                      help='Chest region-type pairs. Should be \
                      specified as a common-separated list of string values \
                      indicating what region-type pairs over which distance was \
                      computed. For a given pair, the first entry is \
                      interpreted as the chest region, and the second entry \
                      is interpreted as the chest type. E.g. LeftLung,Vessel \
                      would be a valid entry.',
                      dest='pair', metavar='<string>', default=None)
    parser.add_option('--out_csv',
                      help='Output csv file with the features', dest='out_csv', 
                      metavar='<string>', default=None)            
    parser.add_option('--in_csv',
                      help='Input csv file with the existing features', 
                      dest='in_csv', metavar='<string>', default=None)  
    (options, args) = parser.parse_args()
    
    #image_io = ImageReaderWriter()
    distance_map, dm_header = nrrd.read(options.in_distance) 
    if (options.in_lm is not None):
        lm,lm_header = nrrd.read(options.in_lm) 
    else:
         lm = np.ones(np.shape(distance_map))   
    in_patches,in_patches_header = nrrd.read(options.in_patches) 
    
    if (options.in_csv is not None):
        # read dataframe from csv
        init_df = pd.read_csv(options.in_csv)
    else:    
        init_df = None

    if options.pair is not None:
        tmp = options.pair.split(',')
        assert len(tmp) == 2, 'Specified pairs not understood'
        pair = [options.pair.split(',')[0],options.pair.split(',')[1] ]
                
    dist_extractor = DistanceFeatureExtractor(options.chest_region, \
        options.chest_type, pair, init_df)                 
    dist_extractor.fit(distance_map, lm, in_patches)

    if options.out_csv is not None:
        dist_extractor.df_.to_csv(options.out_csv, index=False)
        
