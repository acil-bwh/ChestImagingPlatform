import numpy as np
from cip_python.common import ChestConventions

import pandas as pd
import pdb
from cip_python.phenotypes.fissure_phenotypes import FissurePhenotypes

def test_main():
    # First create a test image
    conventions = ChestConventions()
    LUL = conventions.GetChestRegionValueFromName('LeftSuperiorLobe')
    LLL = conventions.GetChestRegionValueFromName('LeftInferiorLobe')
    F = conventions.GetChestTypeValueFromName('ObliqueFissure')
    LLL_F = conventions.GetValueFromChestRegionAndType(LLL, F)
    test_im = LLL*np.ones([1, 8, 8])
    test_im[0, 0, 2::] = LUL
    test_im[0, 1, 2::] = LUL
    test_im[0, 2, 2::] = LUL
    test_im[0, 3, 3::] = LUL
    test_im[0, 4, 4::] = LUL
    test_im[0, 5, 5::] = LUL
    test_im[0, 6, 5::] = LUL
    test_im[0, 7, 5::] = LUL

    test_im[0, 0, 1] = LLL_F
    test_im[0, 1, 1] = LLL_F
    test_im[0, 6, 1] = LLL_F
    test_im[0, 7, 2] = LLL_F
    
    completeness_phenos = FissurePhenotypes()
    df = completeness_phenos.execute(test_im, np.array([1, 1, 2]), 'foo')

    assert df.Completeness.values[0] < 0.5, \
      "Completeness measure should be less than 0.5"

    df = completeness_phenos.execute(test_im, np.array([1, 1, 2]), 'foo',
                                     completeness_type='domain')

    assert df.Completeness.values[0] == 0.5, \
      "Completeness measure should equal 0.5"
    
    test_im = LLL*np.ones([1, 8, 8])
    test_im[0, 0, 2::] = LUL
    test_im[0, 1, 2::] = LUL
    test_im[0, 2, 2::] = LUL
    test_im[0, 3, 3::] = LUL
    test_im[0, 4, 4::] = LUL
    test_im[0, 5, 5::] = LUL
    test_im[0, 6, 5::] = LUL
    test_im[0, 7, 5::] = LUL

    test_im[0, 2, 1] = LLL_F
    test_im[0, 3, 1] = LLL_F
    test_im[0, 4, 1] = LLL_F
    test_im[0, 5, 2] = LLL_F

    df = completeness_phenos.execute(test_im, np.array([1, 1, 2]), 'foo')

    assert df.Completeness.values[0] > 0.5, \
        "Completeness measure should be greater than 0.5"    

    df = completeness_phenos.execute(test_im, np.array([1, 1, 2]), 'foo',
                                     completeness_type='domain')

    assert df.Completeness.values[0] == 0.5, \
      "Completeness measure should equal 0.5"        
