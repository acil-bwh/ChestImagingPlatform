import os.path
import pandas as pd
import nrrd

import numpy as np
import pdb
from pandas.util.testing import assert_frame_equal
import sys

sys.path.append("/Users/rolaharmouche/ChestImagingPlatform/")


from cip_python.classification.distance_feature_extractor_from_ROI \
  import DistExtractorFromROI
      
np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'

def test_execute():
    lm = np.ones([2, 2, 1])
    
    dist_map = np.zeros([2, 2, 1])
    dist_map[0, 0, 0] = 1
    dist_map[0, 1, 0] = 2
    dist_map[1, 0, 0] = 3
    dist_map[1, 1, 0] = 4            
    
    dist_extractor = DistExtractorFromROI(chest_region="WholeLung")
    dist_extractor.fit(dist_map, lm)       


    assert dist_extractor.distance_feature_name =='WholeLungDistance',\
        "Distance feature name not as extected"
    assert dist_extractor.dist_ == 2.5, \
      "WholeLungDistance not as expected"                  

    