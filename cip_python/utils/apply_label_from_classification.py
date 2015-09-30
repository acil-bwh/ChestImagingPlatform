import numpy as np
from optparse import OptionParser
from cip_python.ChestConventions import ChestConventions
import nrrd
import pdb
import pandas as pd
#from cip_python.io.image_reader_writer import ImageReaderWriter
          
def apply_label_from_classification(segmentation, lm, classified_features_df):
        """Given a patch segmentation file, a label map file, and a 
        classification csv file, apply the classification labels to each of 
        the patches in areas where labelmap > 0. The labels follow cip 
        conventions.

        Parameters
        ----------
        segmentation :

        lm :

        classified_features_df :

        Returns
        -------
        classified_lm : 
                
        """    
        conventions = ChestConventions()
        classified_lm = np.zeros(np.shape(segmentation), dtype = 'int') 
    
        #classified_features_df['ChestRegion'].values
        
        patch_labels = classified_features_df['patch_label'].values
        class_types_list = classified_features_df['ChestType'].values
        class_regions_list = classified_features_df['ChestRegion'].values

        chest_types_to_values_map = {}
        chest_regions_to_values_map = {}
        for e in set(class_types_list):
            chest_types_to_values_map[e] = \
              conventions.GetChestTypeValueFromName(e) 

        for e in set(class_regions_list):
            chest_regions_to_values_map[e] = \
              conventions.GetChestRegionValueFromName(e)               
        
        num_feature_vecs = len(classified_features_df)
        for row in range(0, num_feature_vecs):
            type_val = chest_types_to_values_map[class_types_list[row]]
            region_val = chest_regions_to_values_map[class_regions_list[row]]
                
            classified_lm[np.logical_and(segmentation == \
                                         int(patch_labels[row]), lm >0)] = \
                conventions.GetValueFromChestRegionAndType(region_val, type_val) 
        
        return classified_lm

if __name__ == "__main__":
    desc = """Classifies CT images into emphysema subtyoes"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_lm',
                      help='Input mask file. The classification will only be \
                      computed in areas where the mask is > 0. ', 
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--in_patch',
                      help='Input patch file.', 
                      dest='in_patch', metavar='<string>', default=None)
    parser.add_option('--in_csv',
                      help='Input csv file with a classification for every \
                        patch ', dest='in_csv', 
                      metavar='<string>', default=None)          
    parser.add_option('--out_label',
                      help='Output label file with the emphysema \
                        classifications ', dest='out_label', 
                      metavar='<string>', default=None)                                         
                                      
    (options, args) = parser.parse_args()
    
    assert(options.in_csv is not None), "csv input missing"
    assert(options.in_lm is not None), "lm input missing"       

    print "Reading mask..." 
    lm_array, lm_header = nrrd.read(options.in_lm) 
    patch_array, patch_header = nrrd.read(options.in_patch) 

    dataLists = pd.read_csv(options.in_csv)  
   
    labels_array = apply_label_from_classification(patch_array, lm_array, \
        dataLists)

    assert(options.out_label is not None), \
        " outputs missing"   
        
    #print "Writing output labels "
    #nrrd.write(options.out_label, labels_array, lm_header , True, True)
                               
