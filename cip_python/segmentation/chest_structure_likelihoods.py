import pdb

class Likelihoods:
    """Base class for representing likelihood terms for various chest
    structures of interest.

    Parameters
    ----------
    TODO:
        
    """
    def __init__(self):
      pass


class ExponentialLikelihoods:
    """Inhereted class for represening the lung likelihood term with a 2D
    exponential function (features being intensity and distance).
    
    Parameters
    ----------
    w : array, shape ( D, 1 )
        Weight vector for weighting components of feature vectors
        
    lambda : float
        Exponential distribution parameter
        
    feature_vectors: list of feature arrays, shape (L, M, N)
    
    """
    
    def __init__(self, feature_vectors, exp_lambda, coefficients, poly_feature_map):
        self.coefficients = coefficients
        self.feature_vectors = feature_vectors
        self.exp_lambda = exp_lambda
        self.poly_feature_map = poly_feature_map
        
        if poly_feature_map is NULL:
            self.poly_feature_map.feature_vectors = feature_vectors
            self.poly_feature_map.feature_vectors.num_terms = feature_vectors.len
            
        assert coefficients.shape ==  self.poly_feature_map.feature_vectors.num_terms
        
    def compute(self):
        accum = 0
        for d in range(0,self.poly_feature_map.num_terms):
            accum=accum+self.coefficients[d]*self.feature_vectors[d]
            
        return accum
        
        

        
    
