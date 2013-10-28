import pdb

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
        WeightedFeatureMapDensity.__init__(im_feature_vecs, weights, feature_map)
        self.lamda = lamda
        
        if poly_feature_map is None:
            self.poly_feature_map.feature_vectors = feature_vectors
            self.poly_feature_map.feature_vectors.num_terms = feature_vectors.len
            
        assert coefficients.shape ==  self.poly_feature_map.feature_vectors.num_terms
        
    def compute(self):
        accum = 0
        for d in range(0, self.poly_feature_map.num_terms):
            accum = accum + self.coefficients[d]*self.feature_vectors[d]
            
        return accum
        
        

        
    
