import time
from optparse import OptionParser

import nrrd
import numpy as np
import pandas as pd
from scipy import ndimage

from ..common import ChestConventions

class DistanceFeatureExtractor:
    """General purpose class implementing a distance feature extractor. 

    The user inputs distance numpy array, a mask, and a patch segmentation. The
    output is a pandas dataframe with patch #=number and
    a associated distance entries
       
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

    in_df: Pandas dataframe
        Contains feature information previously computed over the patches
        for which we seak the distance information    

    x_extent: int
        region size in the x direction over which the feature will
        be estimated. The region will be centered at the patch center.
            
    y_extent: int
        region size in the y direction over which the feature will
        be estimated. The region will be centered at the patch center.
        
    z_extent: int
        region size in the z direction over which the feature will
        be estimated. The region will be centered at the patch center.
 
                
    Attribues
    ---------
    df_ : Pandas dataframe
        Contains the patch feature information. The 'patch_label' column
        corresponds to the segmented patch over which the distance is 
        computed. The 'chestwall_distance' column contains the physical distance 
        from the center of the patch to the chest wall.        
    """
    def __init__(self, x_extent = 31, y_extent=31, z_extent=1, \
        chest_region=None, chest_type=None, pair=None, in_df=None):
        # get the region / type over which distance is computed

        self.x_half_length = int(np.floor(x_extent/2))
        self.y_half_length = int(np.floor(y_extent/2))
        self.z_half_length = int(np.floor(z_extent/2))
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
        self.df_ = in_df
        
    
    def _create_dataframe(self, patch_labels_for_df=None, in_data = None):
        """Create a pandas dataframe for the computed per patch features, where \
        each patch corresponds to an xml point. 
        
        Parameters
        ----------
        patch_labels_for_df: numpy array, shape (L*M*N)
            Input unique patch segmentation values . Required
        
        in_data:  numpy array, shape (L*M*N)
            Input distance features, wach corresponding to a unique patch value.
            Required     
        """   
         # Initialize the dataframe
        if self.df_ is None:
            cols = ['patch_label', 'ChestRegion', 'ChestType', \
                    self.distance_feature_name]
        else:         
            cols = [self.distance_feature_name]     
          
        num_patches = np.shape(patch_labels_for_df)[0]    
        the_index = np.arange(0,num_patches)
        
        new_df = pd.DataFrame(columns=cols, index = the_index)  
        
        new_df['patch_label'] = patch_labels_for_df  
        new_df = new_df.astype(np.float64)                                      

        new_df.ix[new_df['patch_label'] == patch_labels_for_df, \
            self.distance_feature_name] = in_data                                
        
        if self.df_ is not None:        
            self.df_ = pd.merge(new_df, self.df_, on='patch_label')
        else:
            new_df['ChestRegion'] = 'UndefinedRegion'
            new_df['ChestType'] = 'UndefinedType' 
            self.df_ = new_df
    
                                                                                                                                                                                                                                                                                    
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
        
        assert ((self.x_half_length*2 <= np.shape(distance_image)[0]) and \
            (self.y_half_length*2 <= np.shape(distance_image)[1]) and \
            (self.z_half_length*2 <= np.shape(distance_image)[2])), \
            "distanct region extent must  be less that image dimensions."
                                           
        unique_patch_labels = np.unique(patch_labels[:])
        unique_patch_labels_positive = unique_patch_labels[unique_patch_labels>0]

        patch_center_temp = ndimage.measurements.center_of_mass(patch_labels, \
            patch_labels.astype(np.int32), unique_patch_labels_positive)
            
        # Figure out the number of rows required
        if lm is not None:           
            inc=0
            inc2=0
            p_labels_for_df = np.zeros(np.shape(unique_patch_labels_positive))
            for p_label in unique_patch_labels_positive:
                patch_center = map(int, patch_center_temp[inc2])
                inc2 = inc2+1
                xmin = max(patch_center[0]-self.x_half_length,0)
                xmax =  min(patch_center[0]+self.x_half_length+1,np.shape(lm)[0])
                ymin = max(patch_center[1]-self.y_half_length,0)
                ymax = min(patch_center[1]+\
                    self.y_half_length+1,np.shape(lm)[1])
                zmin = max(patch_center[2]-self.z_half_length,0)
                zmax = min(patch_center[2]+\
                    self.z_half_length+1,np.shape(lm)[2])
                               
                lm_temp = lm[xmin:xmax, ymin:ymax, zmin:zmax]
                if (np.sum(lm_temp>0) > 1):
                    p_labels_for_df[inc] = p_label
                    inc = inc+1
            p_labels_for_df = p_labels_for_df[p_labels_for_df>0]
        else:
            p_labels_for_df = unique_patch_labels[unique_patch_labels>0]

                    
        # loop through each patch 
        inc = 0
        inc2=0
        
        distances = np.zeros(np.shape(p_labels_for_df)[0])
        distance_patch_labels = np.zeros( np.shape(p_labels_for_df)[0], dtype = int)
        
        for p_label in unique_patch_labels_positive:
                tic = time.clock() 
                # extract the lung area from the CT for the patch
                #patch_distances = distance_image[patch_labels==p_label] 
                   
                patch_center = map(int, patch_center_temp[inc])
                
                xmin = max(patch_center[0]-self.x_half_length,0)
                xmax =  min(patch_center[0]+self.x_half_length+1,np.shape(distance_image)[0])
                ymin = max(patch_center[1]-self.y_half_length,0)
                ymax = min(patch_center[1]+\
                    self.y_half_length+1,np.shape(distance_image)[1])
                zmin = max(patch_center[2]-self.z_half_length,0)
                zmax = min(patch_center[2]+\
                    self.z_half_length+1,np.shape(distance_image)[2])
                
                distances_temp = distance_image[xmin:xmax, ymin:ymax, zmin:zmax]
                lm_temp = lm[xmin:xmax, ymin:ymax, zmin:zmax]   
                # extract the lung area from the CT for the patch
                                                 
                patch_distances = distances_temp[(lm_temp >0)] 
                
                if (np.shape(patch_distances)[0] > 1):
                    # linearize features
                    distance_vector = np.array(patch_distances.ravel()).T
                    tic = time.clock() 
                    #inc = inc+1
                    # compute the average distance
                    mean_dist = np.mean(distance_vector) 

                    distances[inc2] = mean_dist
                    distance_patch_labels[inc2]= p_label
                    inc2 = inc2+1  

                inc = inc+1 
                toc = time.clock() 
 
                if np.mod(inc,100) ==0:
                    print("distance "+str(inc)+ " time = "+str(toc-tic))      
        print("creating dataframe")    
        self._create_dataframe(patch_labels_for_df=distance_patch_labels, in_data = distances)      

if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_dist',
                      help='Input distance map file', dest='in_dist', \
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
                      help='Chest region indicating what structure the \
                      distance map corresponds to. Note that either a \
                      region, or a type, or a region-type pair must be \
                      specified.',
                      dest='chest_region', metavar='<string>', default=None)
    parser.add_option('-t',
                      help='Chest type indicating what structure the distance \
                      map corresponds to. Note that either a region, or a \
                      type, or a region-type pair must be specified.',
                      dest='chest_type', metavar='<string>', default=None)
    parser.add_option('-p',
                      help='Chest region-type pair (comma-separated tuple) \
                      indicating what structure the distance map corresponds \
                      to. Note that either a region, or a type, or a \
                      region-type pair must be specified.',
                      dest='pair', metavar='<string>', default=None)
    parser.add_option('--out_csv',
                      help='Output csv file with the features', dest='out_csv', 
                      metavar='<string>', default=None)            
    parser.add_option('--in_csv',
                      help='Input csv file with the existing features. The \
                      distance features will be appended to this.', 
                      dest='in_csv', metavar='<string>', default=None)  
    (options, args) = parser.parse_args()
    
    #image_io = ImageReaderWriter()
    if options.in_dist is None:
        raise ValueError("Must specify as distance map")

    if options.in_patches is None:
        raise ValueError("Must specify a patches segmentation file")
    
    print ("Reading distance map...")
    distance_map, dm_header = nrrd.read(options.in_dist) 

    print ("Reading patches segmentation...")
    in_patches, in_patches_header = nrrd.read(options.in_patches) 
    
    if (options.in_lm is not None):
        print ("Reading mask...")
        lm,lm_header = nrrd.read(options.in_lm) 
    else:
         lm = np.ones(np.shape(distance_map))   
    
    if (options.in_csv is not None):
        print ("Reading previously computed features...")
        init_df = pd.read_csv(options.in_csv)
    else:    
        init_df = None

    if options.chest_region is None and options.chest_type is None and \
      options.pair is None:
      raise ValueError("Must specify a chest region, or chest type, or \
                        region-type pair that the distance map corresponds to")

    pair = None
    if options.pair is not None:
        tmp = options.pair.split(',')
        assert len(tmp) == 2, 'Specified pairs not understood'
        pair = [options.pair.split(',')[0],options.pair.split(',')[1] ]

    print ("Computing distance features...")
    dist_extractor = DistanceFeatureExtractor(options.chest_region, \
        options.chest_type, pair, init_df)                 
    dist_extractor.fit(distance_map, lm, in_patches)

    if options.out_csv is not None:
        print ("Writing...")
        dist_extractor.df_.to_csv(options.out_csv, index=False)

    print ("DONE.")
        
