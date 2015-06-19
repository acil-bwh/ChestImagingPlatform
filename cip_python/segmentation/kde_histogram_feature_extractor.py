import scipy.io as sio
import numpy as np
from optparse import OptionParser
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
import pdb
from kde_bandwidth import botev_bandwidth
import pandas as pd
#from cip_python.io.image_reader_writer import ImageReaderWriter
     
class kdeHistExtractor:
    """General purpose class implementing a kernel density estimation histogram
        extractor. 

    The user inputs CT numpy array, a mask, and a patch segmentation. The
    output is a pandas dataframe with patch #=number and
    1 entry for each histogram bin
       
    Parameters 
    ----------
    num_bins: int
        Number of bins to be produced
    
    lower_limit:int
        lower limit of histogram
        
    upper_limit:int
        upper limit of histogram
    
    in_patches: 3D numpy array, shape (L, M, N)
        Input patch labelmap
        
    in_ct: 3D numpy array, shape (L, M, N)
        Input CT image to be segmentaed
    
    in_lm: 3D numpy array, shape (L, M, N)
        Input mask where histograms will be computed.         
        

    Returns
    --------
    kde_histogram: pandas dataframe
        
    """



    def __perform_kde_botev__(self, input_data):
        """
        perform kernel density estimation  (using botev estimatir for best bandwidth) 
    
        Parameters 
        ----------
        input_data: 1D numpy array
            Array intensity values for which to perform kde
        
        Returns
        --------
        kde_histogram: pandas dataframe
            
        """
        
        input_data = np.squeeze(input_data)

        kde_bandwidth_estimator = botev_bandwidth()
        the_bandwidth = kde_bandwidth_estimator.run(input_data)
        kde = KernelDensity(bandwidth=the_bandwidth)
        kde.fit(input_data[:, np.newaxis])

        
        """ get histogram  (Raul limit to 4096 samples from -1050 till 3050) """
        X_plot = self.bin_values[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        dens = np.exp(log_dens)
        print(np.sum(dens))
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
        ax.fill(self.bin_values, np.exp(log_dens), fc='#AAAAFF')
        plt.show()       
        return dens        
                        
    def __init__(self, lower_limit=-1050, upper_limit=3050, num_bins= 4096):        
        # initialize the dataframe       
        # first get the list of all histogaram bin values
        cols = ['patch_label']
        self.bin_values = np.linspace(lower_limit, upper_limit, num_bins)
        
        for i in self.bin_values:
            cols.append(str(i))
            
        self._df = pd.DataFrame(columns=cols)
        
        
    def execute(self, ct, lm, patch_labels):
        """ compute the histogram
        
        Parameters
        ----------
        patch_labels: 3D numpy array, shape (L, M, N)
            Input patch labelmap
        
        ct: 3D numpy array, shape (L, M, N)
            Input CT image to be segmentaed
    
        lm: 3D numpy array, shape (L, M, N)
            Input mask where histograms will be computed.    
        """
        
        unique_patch_labels = np.unique(patch_labels)
        """ loop through each patch """
        for p_label in unique_patch_labels:               
                print("PATCH LABEL "+str(p_label))
                """ extract the lung area from the CT for the patch"""
                patch_intensities = ct[np.logical_and(patch_labels==p_label, lm >0)] 
        
                """ linearize features"""
                intensity_vector = np.array(patch_intensities.ravel()).T
                if (np.shape(intensity_vector)[0] > 1):
                    print(np.shape(intensity_vector)[0] )
                    """ obtain kde histogram from features"""
                    histogram = self.__perform_kde_botev__(intensity_vector)
                    tmp = dict()
                    tmp['patch_label'] = p_label
                    for i in range(0,np.shape(self.bin_values)[0]):
                        tmp[self.bin_values[i]]= histogram[i]
                    """ save in data frame """
                    self._df = self._df.append(tmp, ignore_index=True)
        return self._df
        

if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file', dest='in_ct', metavar='<string>',
                      default=None)
    parser.add_option('--in_lm',
                      help='Input mask file. The histogram will only be \
                      computed in areas where the mask is > 0', 
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--in_patches',
                      help='Input patch labels file. A label is defined for \
                      each corresponding CT voxel. A histogram will be \
                      computed for each patch label', dest='in_patches', 
                      metavar='<string>',default=None)
    parser.add_option('--out_csv',
                      help='Output csv file with the features', dest='out_csv', 
                      metavar='<string>', default=None)          
    parser.add_option('--num_bins',
                      help='Number of histogram bins.  (optional)',  dest='num_bins', 
                      metavar='<string>', default=4096)    
    parser.add_option('--lower_limit',
                      help='lower histogram limit.  (optional)',  dest='lower_limit', 
                      metavar='<string>', default=-1050)                        
    parser.add_option('--upper_limit',
                      help='upper histogram limit.  (optional)',  dest='upper_limit', 
                      metavar='<string>', default=3050)  
    (options, args) = parser.parse_args()
    
    #image_io = ImageReaderWriter()
    import nrrd
    ct,ct_header=  nrrd.read(options.in_ct) #image_io.read_in_numpy(options.in_ct)
    lm,lm_header=  nrrd.read(options.in_lm) 
    in_patches,in_patches_header=  nrrd.read(options.in_patches) 
    
    kde_hist_extractor = kdeHistExtractor(lower_limit=options.lower_limit, \
        upper_limit=options.upper_limit, num_bins=options.num_bins)
                 
    kde_hist_datafram = kde_hist_extractor.execute(ct, lm, in_patches)

    if options.out_csv is not None:
        kde_hist_datafram.to_csv(options.out_csv, index=False)
        
