import sys
from cip_python.utils.feature_maps import PolynomialFeatureMap
import numpy as np
#import Capture

#run nosetests --nocapture in utils directory

I1 = np.array([[[1,2,3,4],[5,6,7,8],[1,2,3,4]],[[0,1,0,2],[2,1,1,3],[6,7,4,5]]])
I2 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])

prior1 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])
prior2 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])
 
I1 = I1/10
I2 = I2 / 100

prior1=prior1/120
prior2=prior2/115

def test_segment():
    sys.stderr.write(str(I1)+"\n")
    sys.stderr.write(str(I2)+"\n")

    my_polynomial_feature_map = PolynomialFeatureMap( [I1],[1,2] )  
    my_polynomial_feature_map.compute_num_terms()
    first_element = my_polynomial_feature_map.get_mapped_feature_vec_element(2)
    result = np.multiply(I1,I1)
    assert (first_element == result).all()
    
    
 