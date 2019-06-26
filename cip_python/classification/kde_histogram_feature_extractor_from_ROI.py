import warnings

import numpy as np
from sklearn.neighbors import KernelDensity

from cip_python.classification import botev_bandwidth

class kdeHistExtractorFromROI:
    """General purpose class implementing a kernel density estimation histogram
    extractor from the CT patch. 

    The user inputs CT patch and a labelmap mask. The output is an array with 
    1 entry for each histogram bin. 
       
    Parameters 
    ----------    
    lower_limit:int
        Lower limit of histogram
        
    upper_limit:int
        Upper limit of histogram
 
 
                             
    Attribues
    ---------
    bin_values: Numpy array of shape (N,), where N is the number of bins
        Contains the intensity values for which kde was estimated
        
    hist_ : Numpy array of shape (N,), where N is the number of bins
        Contains the computed histogram information. The 'bin_values',  range 
        from 'lower_limit' to 'upper_limit'. These columns record the number of 
        counts for each Hounsfield unit (hu) estimated by the KDE.        
    """
    def __init__(self, lower_limit=-1050, upper_limit=3050):

        #  get the list of all histogaram bin values
        self.bin_values = np.arange(lower_limit, upper_limit+1)
        self.hist_ = None
        self.kde_ = None
            
       
                                                            
    def _perform_kde_botev(self, input_data):
        """Perform kernel density estimation  (using botev estimator for best 
        bandwidth) 
    
        Parameters 
        ----------
        input_data : 1D numpy array
            Array intensity values for which to perform kde
        
        Returns
        --------
        the_hist : array, shape (n_bins)
            The computed histogram
        """  
              
        input_data = np.squeeze(input_data)

        kde_bandwidth_estimator = botev_bandwidth()
        the_bandwidth = kde_bandwidth_estimator.run(input_data)
        
        if (the_bandwidth <= 0):
            warnings.warn("Bandwidth negative: "+str(the_bandwidth)+\
                " Set to 0.02")
            the_bandwidth = 0.02
            
        kde = KernelDensity(bandwidth=the_bandwidth, rtol=1e-6)
        #print(input_data[:, np.newaxis])
        kde.fit(input_data[:, np.newaxis])
        
        # Get histogram 
        X_plot = self.bin_values[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        #print(log_dens)
        the_hist = np.exp(log_dens)
        self.kde_ = kde
        #the_hist = the_hist/np.sum(the_hist)
         
        return the_hist

                
    def fit(self, ct_patch, lm_patch):
        """Compute the histogram of each patch defined in 'patch_labels' beneath
        non-zero voxels in 'lm' using the CT data in 'ct'.
        
        Parameters
        ----------
        
        ct: 3D numpy array, shape (L, M, N)
            Input CT patch from which histogram information will be derived
    
        lm: 3D numpy array, shape (L, M, N)
            Input mask mask histograms will be computed.    
        """                
            
        # extract the lung area from the CT for the patch
        patch_intensities = ct_patch[(lm_patch >0)] 
        # linearize features
        if (np.shape(patch_intensities)[0] > 1):
            intensity_vector = np.array(patch_intensities.ravel()).T
                
            # obtain kde histogram from features
            self.hist_ = self._perform_kde_botev(intensity_vector)[\
                0:np.shape(self.bin_values)[0]]


        if(np.sum(self.hist_)<0.01 and np.sum(self.hist_)>0):
            self.hist_=self.hist_/np.sum(self.hist_)
