import numpy as np

from cip_python.common import Paths
from cip_python.classification import DistExtractorFromROI
      
np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

ct_name = Paths.testing_file_path('simple_ct.nrrd')

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

    