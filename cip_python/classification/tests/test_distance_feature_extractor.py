import numpy as np
from cip_python.classification import DistanceFeatureExtractor


np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

def test_execute():
    lm = np.ones([2, 2, 1])
    patches = np.ones([2, 2, 1])
    
    dist_map = np.zeros([2, 2, 1])
    dist_map[0, 0, 0] = 1
    dist_map[0, 1, 0] = 2
    dist_map[1, 0, 0] = 3
    dist_map[1, 1, 0] = 4            
    
    dist_extractor = DistanceFeatureExtractor(chest_region="WholeLung", \
        x_extent=2, y_extent = 2, z_extent = 1)
    dist_extractor.fit(dist_map, lm, patches)

    assert dist_extractor.df_['patch_label'].values[0] == 1, \
      "patch_label not as expected"

    assert dist_extractor.df_['ChestRegion'].values[0] == 'UndefinedRegion', \
      "ChestRegion not as expected"      

    assert dist_extractor.df_['ChestType'].values[0] == 'UndefinedType', \
      "ChestType not as expected"            

    assert dist_extractor.df_['WholeLungDistance'].values[0] == 2.5, \
      "WholeLungDistance not as expected"                  

    lm = np.ones([2, 2, 2])
    patches = np.ones([2, 2, 2])
    
    # test #2
    
    dist_map = np.zeros([2, 2, 2])
    dist_map[0, 0, 0] = 1
    dist_map[0, 1, 0] = 2
    dist_map[1, 0, 0] = 3
    dist_map[1, 1, 0] = 4            
    dist_map[0, 0, 1] = 2
    dist_map[0, 1, 1] = 3
    dist_map[1, 0, 1] = 4
    dist_map[1, 1, 1] = 5      
       

    dist_extractor = DistanceFeatureExtractor(chest_region="WholeLung", \
        x_extent=2, y_extent = 2, z_extent = 2)
    dist_extractor.fit(dist_map, lm, patches)

    #pdb.set_trace()
    
    assert dist_extractor.df_['patch_label'].values[0] == 1, \
      "patch_label not as expected"

    assert dist_extractor.df_['ChestRegion'].values[0] == 'UndefinedRegion', \
      "ChestRegion not as expected"      

    assert dist_extractor.df_['ChestType'].values[0] == 'UndefinedType', \
      "ChestType not as expected"            

    assert dist_extractor.df_['WholeLungDistance'].values[0] == 3.0, \
      "WholeLungDistance not as expected"                  

