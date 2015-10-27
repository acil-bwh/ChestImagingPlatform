import os.path
import pandas as pd
import nrrd

import numpy as np
import sys
sys.path.append("/Users/rolaharmouche/ChestImagingPlatform/")
import pdb
from pandas.util.testing import assert_frame_equal

from cip_python.classification.get_ct_patch_from_center \
  import get_bounds_from_center
from cip_python.classification.get_ct_patch_from_center \
  import get_patch_given_bounds
    


  
np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'

def test_execute():
    ct_array, ct_header = nrrd.read(ct_name)

    extent = [5,11,1] 
    center = [2,5,1]
    patch_ref = np.squeeze(ct_array[:, :, 1])

    test_bounds = get_bounds_from_center(ct_array,center,extent)
    test_patch = get_patch_given_bounds(ct_array,test_bounds)
    
    pdb.set_trace()
    np.testing.assert_array_equal(test_patch,patch_ref, err_msg='arrays not equal', verbose=True)

