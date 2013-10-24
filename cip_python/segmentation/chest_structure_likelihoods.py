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
    
    """
    
    def __init__(self):
        self.coefficients = []
        
    def set_coefficients(self, list_of_coefficients):
        self.coefficients =  list_of_coefficients
        
