import sys
from cip_python.utils.feature_maps import PolynomialFeatureMap
import numpy as np
from cip_python.utils.weighted_feature_maps import ExpWeightedFeatureMapDensity

I1 = np.array([[[1,2,3,4],[5,6,7,8],[1,2,3,4]],[[0,1,0,2],[2,1,1,3],[6,7,4,5]]])
I2 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])

def test_weighted_features():
    sys.stderr.write(str(I1)+"\n")
    sys.stderr.write(str(I2)+"\n")

    my_polynomial_feature_map = PolynomialFeatureMap( [I1],[1,2] )  
    my_polynomial_feature_map.compute_num_terms()
