import scipy.io as sio
import numpy as np
from optparse import OptionParser
import matplotlib.pyplot as plt
import pdb
import pandas as pd
import warnings
from sklearn.neighbors import KernelDensity
import nrrd
from kde_bandwidth import botev_bandwidth

class kde_lattice_classifier:
    """General purpose class implementing a lattice kernel density classification. 

    The user inputs a list of dataframe with kde, distance and neighbourhood 
    information (one df per patch). Each patch  is segmented into one of the 
    predefined classes. The output is a dataframe with a class label for patch 
    ID.
       
    Parameters 
    ----------    
    kde_lower_limit:int
        Lower limit of kde histogram
        
    kde_upper_limit:int
        Upper limit of kde histogram
        
    Attribues
    ---------
    _patch_labels_df : pandas dataframe
        Contains the class labels corresponding to the input patch ids
    """
    
    def __init__(self, kde_lower_limit, kde_upper_limit):
        pass
        
    def fit(self, input_kde, input_distance, input_neighbours):
        pass