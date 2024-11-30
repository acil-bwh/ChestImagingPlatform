import sys
from cip_python.utils import PolynomialFeatureMap
import numpy as np
#import Capture

#run ${CIP_NOSETESTS_EXEC} --nocapture in utils directory

I1 = np.array([[[1,2,3,4],[5,6,7,8],[1,2,3,4]],[[0,1,0,2],[2,1,1,3],[6,7,4,5]]])
I2 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])
I3 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])

def test_polynomial_feature_map_num_terms():
    my_polynomial_feature_map = PolynomialFeatureMap([I1, I2], [1, 2])  
    my_polynomial_feature_map.compute_num_terms()
    sys.stdout=my_polynomial_feature_map.num_terms
    assert my_polynomial_feature_map.num_terms == 5
    
    my_polynomial_feature_map = PolynomialFeatureMap( [I1,I2], [0])  
    my_polynomial_feature_map.compute_num_terms()
    sys.stdout=my_polynomial_feature_map.num_terms
    assert my_polynomial_feature_map.num_terms == 1    

    my_polynomial_feature_map = PolynomialFeatureMap([I1,I2], [2,2])  
    my_polynomial_feature_map.compute_num_terms()
    sys.stdout=my_polynomial_feature_map.num_terms
    assert my_polynomial_feature_map.num_terms == 3   
        
    my_polynomial_feature_map = PolynomialFeatureMap([I1,I2], [2,1,2])  
    my_polynomial_feature_map.compute_num_terms()
    sys.stdout=my_polynomial_feature_map.num_terms
    assert my_polynomial_feature_map.num_terms == 5    
    
    my_polynomial_feature_map = PolynomialFeatureMap([I1,I2,I3], [2])  
    my_polynomial_feature_map.compute_num_terms()
    sys.stdout=my_polynomial_feature_map.num_terms
    assert my_polynomial_feature_map.num_terms == 6
    
    my_polynomial_feature_map = PolynomialFeatureMap([I1,I2,I3], [2,1,2,1,1])  
    my_polynomial_feature_map.compute_num_terms()
    sys.stdout=my_polynomial_feature_map.num_terms
    assert my_polynomial_feature_map.num_terms == 9  
 
def test_polynomial_feature_map():
    sys.stderr.write(str(I1)+"\n")
    sys.stderr.write(str(I2)+"\n")

    my_polynomial_feature_map = PolynomialFeatureMap( [I1,I2],[1,2] )  
    my_polynomial_feature_map.compute_num_terms()
    first_element = my_polynomial_feature_map.get_mapped_feature_vec_element(2)
    result = np.multiply(I1,I1)
    assert (first_element == result).all()
    
    first_element = my_polynomial_feature_map.get_mapped_feature_vec_element(0)
    result = I1
    assert (first_element == result).all()
    
    my_polynomial_feature_map = PolynomialFeatureMap( [I1,I2], [0])  
    my_polynomial_feature_map.compute_num_terms()
    first_element = my_polynomial_feature_map.get_mapped_feature_vec_element(0)
    sys.stderr.write(str(first_element)+"\n")
    result = 1
    assert (first_element == result)   

    my_polynomial_feature_map = PolynomialFeatureMap([I1,I2], [2])  
    my_polynomial_feature_map.compute_num_terms()
    first_element = my_polynomial_feature_map.get_mapped_feature_vec_element(1)
    result = np.multiply(I1,I2)
    assert (first_element == result).all() 
        
    my_polynomial_feature_map = PolynomialFeatureMap([I1,I2], [2,2,1,2])  
    my_polynomial_feature_map.compute_num_terms()
    first_element = my_polynomial_feature_map.get_mapped_feature_vec_element(1)
    result = I2
    assert (first_element == result).all()   

  
