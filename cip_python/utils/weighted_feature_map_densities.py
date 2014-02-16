import pdb
import numpy as np

class WeightedFeatureMapDensity:
    """Base class for representing likelihood terms for various chest
    structures of interest.

    The classes derived from this class provide functionality to take data
    feature vectors, map them to some other space according to the supplied
    mapping, weight the feature vector in the mapped space, and then compute
    probability density values for those weighted, mapped, feature vectors.
    This usefulness of this class is apparent when dealing with large data
    inputs (e.g. images) and with large dimensions in the mapped space: this
    class attempts to compute density values in a memory and speed efficient
    manner.

    Parameters
    ----------
    im_feature_vecs : list of feature images
        Each element of the list is a scalar image (all images must have the
        same dimension and size) representing a given dimension in the input
        feature space. For example, one image may be HU intensity values,
        another may be distance values to some structure, etc.    
    
    weights : array, shape ( D, 1 )
        Weight vector for weighting components of mapped feature vectors. If no
        feature mapping is specified, the weights will be assumed to apply to
        the input feature vectors.

    feature_map : type FeatureMap, optional
        Instance of FeatureMap which indicates how the input feature vectors
        should be mapped. If no feature map is specified, the specified weight
        vector will be assumed to apply to the input feature vectors.    
    """
    def __init__(self, im_feature_vecs, weights, feature_map):
        self.im_feature_vecs = im_feature_vecs
        self.weights = weights
        self.feature_map = feature_map

class ExpWeightedFeatureMapDensity(WeightedFeatureMapDensity):
    """Weighted feature map density for the exponential distribution.
    
    Parameters
    ----------
    im_feature_vecs : list of feature images
        Each element of the list is a scalar image (all images must have the
        same dimension and size) representing a given dimension in the input
        feature space. For example, one image may be HU intensity values,
        another may be distance values to some structure, etc.    
    
    weights : array, shape ( D, 1 )
        Weight vector for weighting components of mapped feature vectors. If no
        feature mapping is specified, the weights will be assumed to apply to
        the input feature vectors.

    feature_map : type FeatureMap, optional
        Instance of FeatureMap which indicates how the input feature vectors
        should be mapped. If no feature map is specified, the specified weight
        vector will be assumed to apply to the input feature vectors.
        
    lamda : float
        Exponential distribution parameter            
    """
    
    def __init__(self, im_feature_vecs, weights, feature_map, lamda):
        WeightedFeatureMapDensity.__init__(self,im_feature_vecs, weights, \
           feature_map) 
        self.lamda = lamda
        
        if feature_map is None:
            self.feature_map.feature_vectors = im_feature_vecs.astype(np.float)
            self.feature_map.num_terms = im_feature_vecs.len
            
        assert len(self.weights) ==  self.feature_map.num_terms
        
    def compute(self):
        accum = \
            self.weights[0]*self.feature_map.get_mapped_feature_vec_element(0)
        for d in range(1, self.feature_map.num_terms):  
            accum = accum + \
              self.weights[d]*self.feature_map.get_mapped_feature_vec_element(d)
        exponential_density = np.exp(-self.lamda*accum)*self.lamda
        
        return exponential_density
        
   


        
    
