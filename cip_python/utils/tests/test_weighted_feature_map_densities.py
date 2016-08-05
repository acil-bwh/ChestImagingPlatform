import sys
from cip_python.utils import PolynomialFeatureMap
import numpy as np
from cip_python.utils import ExpWeightedFeatureMapDensity

I1 = np.array([[[1,2,3,4],[5,6,7,8],[1,2,3,4]],[[0,1,0,2],[2,1,1,3],[6,7,4,5]]])
I2 = np.array([[[3,2,6,2],[4,5,9,100],[10,23,32,2]],[[1,5,2,4],[5,7,2,6],[8,2,3,7]]])

l_alpha_est_left=[0.002149, -0.002069, 5.258745]
l_alpha_est_right=[0.001241, -0.025153, 4.609616]
l_alpha_est_non=[-0.001929, 0.010123, 3.937502]  
    
def test_weighted_features():
    sys.stderr.write(str(I1)+"\n")
    sys.stderr.write(str(I2)+"\n")

    my_polynomial_feature_map = PolynomialFeatureMap( [I1,I2],[2] )  
    my_polynomial_feature_map.compute_num_terms()
    
    assert my_polynomial_feature_map.num_terms == 3

    #define the weights
    
    #exp(-(alpha_est_non(1)*Ival + alpha_est_non(2)*Dval+alpha_est_non(3)))^2
    
    the_weights = [1.0, 0.0, 0.0 ]
    the_lambda = 1.0
    #(self, im_feature_vecs, weights, feature_map, lamda)
    my_weighted_density = ExpWeightedFeatureMapDensity([I1,I2], the_weights, my_polynomial_feature_map, the_lambda)
    my_likelihood = my_weighted_density.compute()
    
    assert my_likelihood.all() == np.exp(-np.power(I1,2)).all()
    

    the_weights = [0.0, 1.0, 0.0 ]
    
    my_weighted_density = ExpWeightedFeatureMapDensity([I1,I2], the_weights, my_polynomial_feature_map, the_lambda)
    my_likelihood = my_weighted_density.compute()
    
    assert my_likelihood.all() == np.exp(-np.power(I2,2)).all()
    
    the_weights = [0.3, 0.5, 1.3 ]
    
    my_weighted_density = ExpWeightedFeatureMapDensity([I1,I2], the_weights, my_polynomial_feature_map, 1.0)
    my_likelihood = my_weighted_density.compute()
    
    assert my_likelihood.all() == np.exp(-(the_weights[0]*np.power(I1,2) + the_weights[1]*I1*I2+ the_weights[2]*np.power(I2,2)) ).all()

def test_weighted_features_elaborate():
    #Slightly more elaborate, similar to likelihood computations needed for
    # lung segmentation
    left_polynomial_feature_map = PolynomialFeatureMap( [I1,I2],[0,1,2] )  
    left_polynomial_feature_map.compute_num_terms()
    
    left_weights_temp = [0.002149, -0.002069, 5.258745]
    left_weights = [left_weights_temp[2]*left_weights_temp[2], 2*left_weights_temp[0]*left_weights_temp[2], \
                    2*left_weights_temp[1]*left_weights_temp[2], left_weights_temp[0]*left_weights_temp[0], \
                    2*left_weights_temp[0]*left_weights_temp[1], left_weights_temp[1]*left_weights_temp[1] ]
    left_lambda = 1.0
    left_weighted_density = ExpWeightedFeatureMapDensity([I1,I2], left_weights, left_polynomial_feature_map, left_lambda)
    left_likelihood = left_weighted_density.compute()
    
    #should be equivelent to: exp(-(alpha_est_non(1)*Ival + alpha_est_non(2)*Dval+alpha_est_non(3)))^2
    
    assert left_likelihood.all() == np.exp(-np.power(   (left_weights_temp[0]*I1 + left_weights_temp[1]*I2+ left_weights_temp[2])   , 2) ).all()
    
    
    
    
    
    
    
    
    