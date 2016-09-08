import numpy as np

from cip_python.classification import DistanceFeatureExtractorFromXML
from cip_python.input_output import ImageReaderWriter
from cip_python.common import Paths


np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

ct_name = Paths.testing_file_path('simple_ct.nrrd')
in_xml = Paths.testing_file_path('simple_ct_regionAndTypePoints.xml')

def test_execute():
    lm = np.ones([5, 11, 3])
    
    dist_map = np.zeros([5, 11, 3])
    dist_map[2, 5, 1] = 10
    dist_map[1, 5, 1] = 2
    dist_map[3, 5, 1] = 3
    dist_map[2, 4, 1] = 9
    dist_map[1, 4, 1] = 4
    dist_map[3, 4, 1] = 5
    dist_map[2, 6, 1] = 6
    dist_map[1, 6, 1] = 7
    dist_map[3, 6, 1] = 8
    
    dist_map[2, 5, 2] = 4
    dist_map[1, 5, 2] = 5
    dist_map[3, 5, 2] = 6
    dist_map[2, 4, 2] = 4
    dist_map[1, 4, 2] = 5
    dist_map[3, 4, 2] = 6
    dist_map[2, 6, 2] = 4
    dist_map[1, 6, 2] = 5
    dist_map[3, 6, 2] = 6  
    
                  
    dist_map[2, 9, 2] = 15    
    dist_map[2, 9, 1] = 20 
    dist_map[2, 5, 0] = 25
             
    image_io = ImageReaderWriter()
    ct_array, ct_header=image_io.read_in_numpy(ct_name)
    
    with open(in_xml, 'r+b') as f:
        xml_data = f.read() # coords 3,5,1 and 3,5,2
        
    dist_extractor = DistanceFeatureExtractorFromXML(chest_region="WholeLung", \
        x_extent=2, y_extent = 2, z_extent = 1)
    dist_extractor.fit(dist_map, ct_header, lm, xml_data)

    #assert dist_extractor.df_['patch_label'].values[0] == 1, \
    #  "patch_label not as expected"

#    assert dist_extractor.df_['ChestRegion'].values[0] == 'UndefinedRegion', \
#      "ChestRegion not as expected"      
#
#    assert dist_extractor.df_['ChestType'].values[0] == 'UndefinedType', \
#      "ChestType not as expected"            

    print(dist_extractor.df_)
    assert dist_extractor.df_['WholeLungDistance'].values[0] == 6, \
      "WholeLungDistance 1 not as expected"                  

    assert dist_extractor.df_['WholeLungDistance'].values[1] == 5, \
      "WholeLungDistance 2 not as expected"   
   
