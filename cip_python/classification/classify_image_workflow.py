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
from knn_classifier import KnnClassifier

class KdeCTClassifier:
    """General purpose class implementing a ct kernel density classification. 

    The user inputs CT numpy array and a mask. A patch array is generated and
    each patch where mask > 0 is segmented into one of the predefined classes.
    The output is a numpy array with a class label for each corresponding CT 
    voxel. To classify 1 patch,  ct array of patch size is input.
       
    Parameters 
    ----------    
    kde_lower_limit:int
        Lower limit of kde histogram
        
    kde_upper_limit:int
        Upper limit of kde histogram

    patch_type:string
        Type of patches to be produced. Options : grid, slic
        
    Attribues
    ---------
    _image_classes : 3D numpy array, shape (L, M, N)
        Contains the class labels corresponding to the input CT image at each 
        patch location.        
    """
    
    def __init__(self, kde_lower_limit=-1050, kde_upper_limit=-450, patch_type="grid", \
            classifier_type = "KNN"):
        self.kde_lower_limit = kde_lower_limit
        self.kde_upper_limit = kde_upper_limit
        self.patch_type = patch_type 
        self.classifier_type = classifier_type 
        
    def fit(self, ct, lm=None, distance_image=None):        
        # compute patches depending on option selected
        if (patch_type == "grid"):
            grid_segmentor = GridSegmentor(input_dimensions=None, ct=ct)
            patch_segmentation = grid_segmentor.execute()

        # get intensity kde for all patches
        my_hist_extractor = kdeHistExtractor()#lower_limit=-1050, upper_limit=50, num_bins= 1100)
        my_hist_extractor.fit( ct, lm, patch_segmentation)
        # my_hist_extractor.df_
        
        # get distance features for all patches 
        if distance_image is not None:
                # get average distance using patch mask
            distance_image = DistanceFeatureExtractor(chest_region='wholeLung')
            my_dist_extractor.fit(distance_image, lm, patch_segmentation)
            thedf_dist = thedf_dist.append(my_dist_extractor.df_, ignore_index=True)
  
        # perform classification
        if (classifier_type == "KNN"):
            my_knn_classifier = KnnClassifier(patch_type="grid")
            my_knn_classifier.fit(my_hist_extractor.df_, thedf_dist.df_)
            #classify_lattice(my_hist_extractor.df_, thedf_dist.df_)
        