import scipy.io as sio
import numpy as np
from optparse import OptionParser
import pdb
import pandas as pd
import warnings
from sklearn.neighbors import KernelDensity
from scipy import ndimage
import nrrd
from kde_bandwidth import botev_bandwidth
#from cip_python.io.image_reader_writer import ImageReaderWriter
import time
     
class kdeHistExtractor:
    """General purpose class implementing a kernel density estimation histogram
    extractor. 

    The user inputs CT numpy array, a mask, and a patch segmentation. The
    output is a pandas dataframe with patch #=number and
    1 entry for each histogram bin. For each patch, the kde will be computed 
    using intensities from a region centered at the patch center. The region
    extent is:
    [(x_center - floor(x_extent/2):(x_center + floor(x_extent/2), \
        y_center - floor(y_extent/2):y_center + floor(y_extent/2) , \
       ( z_center - floor(z_extent/2)): z_center + floor(z_extent/2)]
       
    Parameters 
    ----------    
    lower_limit:int
        Lower limit of histogram
        
    upper_limit:int
        Upper limit of histogram

    in_df: Pandas dataframe
        Contains feature information previously computed over the patches
        for which we seak the distance information    
        
    x_extent: int
        region size in the x direction over which the kde will
        be estimated. The region will be centered at the patch center.
            
    y_extent: int
        region size in the y direction over which the kde will
        be estimated. The region will be centered at the patch center.
        
    z_extent: int
        region size in the z direction over which the kde will
        be estimated. The region will be centered at the patch center.
 
                             
    Attribues
    ---------
    df_ : Pandas dataframe
        Contains the computed histogram information. The 'patch_label' column
        corresponds to the segmented patch over which the histogram is 
        computed. The remaining columns are named 'huNum', where 'Num' ranges 
        from 'lower_limit' to 'upper_limit'. These columns record the number of 
        counts for each Hounsfield unit (hu) estimated by the KDE.        
    """
    def __init__(self, lower_limit=-1050, upper_limit=3050, x_extent = 31, \
        y_extent=31, z_extent=1, in_df=None):
        # Initialize the dataframe       
        # first get the list of all histogaram bin values
        self.bin_values = np.arange(lower_limit, upper_limit+1)
        self.x_half_length = np.floor(x_extent/2)
        self.y_half_length = np.floor(y_extent/2)
        self.z_half_length = np.floor(z_extent/2)
        
        cols = []
        for i in self.bin_values:
            cols.append('hu' + str(int(round(i))))
        
        if in_df is None:
            cols.append('patch_label')
            cols.append('ChestRegion')
            cols.append('ChestType')
                
            self.df_ = pd.DataFrame(columns=cols)
        else:         
            self.df_ = in_df
            for c in cols:
                if c not in self.df_.columns:
                    self.df_[c] = np.nan
        
    def _perform_kde_botev(self, input_data):
        """Perform kernel density estimation  (using botev estimator for best 
        bandwidth) 
    
        Parameters 
        ----------
        input_data : 1D numpy array
            Array intensity values for which to perform kde
        
        Returns
        --------
        hist : array, shape (n_bins)
            The computed histogram
        """  
              
        input_data = np.squeeze(input_data)

        kde_bandwidth_estimator = botev_bandwidth()
        the_bandwidth = kde_bandwidth_estimator.run(input_data)
        
        #if (the_bandwidth < 0.1):
        #    warnings.warn("Bandwidth less than 0.1: "+str(the_bandwidth))
        if (the_bandwidth <= 0):
            warnings.warn("Bandwidth negative: "+str(the_bandwidth)+\
                " Set to 0.02")
            the_bandwidth = 0.02
            
        kde = KernelDensity(bandwidth=the_bandwidth, rtol=1e-6)
        kde.fit(input_data[:, np.newaxis])
        
        # Get histogram 
        X_plot = self.bin_values[:, np.newaxis]
        log_dens = kde.score_samples(X_plot)
        the_hist = np.exp(log_dens)
        #the_hist = the_hist/np.sum(the_hist)
        

        
        return the_hist
        
    def fit(self, ct, lm, patch_labels):
        """Compute the histogram of each patch defined in 'patch_labels' beneath
        non-zero voxels in 'lm' using the CT data in 'ct'.
        
        Parameters
        ----------
        patch_labels: 3D numpy array, shape (L, M, N)
            Input patch segmentation
        
        ct: 3D numpy array, shape (L, M, N)
            Input CT image from which histogram information will be derived
    
        lm: 3D numpy array, shape (L, M, N)
            Input mask where histograms will be computed.    
        """                
        #patch_labels_copy = np.copy(patch_labels)
        #patch_labels_copy[lm == 0] = 0
        
        
        
        unique_patch_labels = np.unique(patch_labels[:])
        unique_patch_labels = unique_patch_labels[unique_patch_labels !=0]

        assert ((self.x_half_length*2 <= np.shape(ct)[0]) and \
            (self.y_half_length*2 <= np.shape(ct)[1]) and \
            (self.z_half_length*2 <= np.shape(ct)[2])), "kde region extent must \
            be less that image dimensions."
        
        # loop through each patch 
        inc = 0
        
        patch_center_temp = ndimage.measurements.center_of_mass(patch_labels, \
                    patch_labels.astype(np.int32), unique_patch_labels)
        for p_label in unique_patch_labels:
                if np.mod(p_label,100) ==0:
                    print("computing histogram for patch "+str(p_label))
                patch_center = map(int, patch_center_temp[inc])
                inc = inc+1

                xmin = max(patch_center[0]-self.x_half_length,0)
                xmax =  min(patch_center[0]+self.x_half_length+1,np.shape(ct)[0])
                ymin = max(patch_center[1]-self.y_half_length,0)
                ymax = min(patch_center[1]+\
                    self.y_half_length+1,np.shape(ct)[1])
                zmin = max(patch_center[2]-self.z_half_length,0)
                zmax = min(patch_center[2]+\
                    self.z_half_length+1,np.shape(ct)[2])
                
                #print("center of mass time: "+str(toc-tic))
                intensities_temp = ct[xmin:xmax, ymin:ymax, zmin:zmax]                    
                lm_temp = lm[xmin:xmax, ymin:ymax, zmin:zmax]
                                   
                #print("lm extraction time: "+str(toc2-toc))
                # extract the lung area from the CT for the patch
                patch_intensities = intensities_temp[(lm_temp >0)] 
                # linearize features
                intensity_vector = np.array(patch_intensities.ravel()).T
                
                if (np.shape(intensity_vector)[0] > 1):
                    # obtain kde histogram from features
                    tic = time.clock()     
                    histogram = self._perform_kde_botev(intensity_vector)
                    
                    index = self.df_['patch_label'] == p_label
                    
                    if np.sum(index) > 0:
                        toc1 = time.clock() 
                        for i in range(0, np.shape(self.bin_values)[0]):
                            self.df_.ix[index, 'hu' + str(self.bin_values[i])] \
                              = histogram[i]
                        toc2 = time.clock() 
                        print("execution time of post hist extractor 1= "+str(toc2 - toc1))   
                    else:      
                        toc1 = time.clock()               
                        tmp = dict()
                        tmp['ChestRegion'] = 'UndefinedRegion'
                        tmp['ChestType'] = 'UndefinedType'
                        tmp['patch_label'] = p_label
                        toc2 = time.clock() 
                        for i in range(0, np.shape(self.bin_values)[0]):
                            tmp['hu' + str(self.bin_values[i])] = histogram[i]
                        toc3 = time.clock() 
                        self.df_ = self.df_.append(tmp, ignore_index=True)
                        toc4 = time.clock() 
                        #import timeit
                        #timeit self.df_.append(tmp, ignore_index=True)
                        #pdb.set_trace()
                    
                        #print("execution time of post hist extractor 2 new 1= "+str(toc2 - toc1))   
                        #print("execution time of post hist extractor 2 new 2= "+str(toc3 - toc2))   
                        #print("execution time of post hist extractor 2 new 3= "+str(toc4 - toc3))   
          
                
if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file', dest='in_ct', metavar='<string>',
                      default=None)
    parser.add_option('--in_lm',
                      help='Input mask file. The histogram will only be \
                      computed in areas where the mask is > 0. If lm is not \
                      included, the histogram will be computed everywhere.', 
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--in_patches',
                      help='Input patch labels file. A label is defined for \
                      each corresponding CT voxel. A histogram will be \
                      computed for each patch label', dest='in_patches', 
                      metavar='<string>',default=None)
    parser.add_option('--in_csv',
                      help='Input csv file with the existing features. The \
                      histogram features will be appended to this.', 
                      dest='in_csv', metavar='<string>', default=None)                        
    parser.add_option('--out_csv',
                      help='Output csv file with the features', dest='out_csv', 
                      metavar='<string>', default=None)            
    parser.add_option('--lower_limit',
                      help='lower histogram limit.  (optional)',  dest='lower_limit', 
                      metavar='<string>', default=-1050)                        
    parser.add_option('--upper_limit',
                      help='upper histogram limit.  (optional)',  dest='upper_limit', 
                      metavar='<string>', default=3050)  
    (options, args) = parser.parse_args()
    
    #image_io = ImageReaderWriter()
    print "Reading CT..."
    ct, ct_header = nrrd.read(options.in_ct) #image_io.read_in_numpy(options.in_ct)
    
    if (options.in_lm is not None):
        print "Reading mask..." 
        lm, lm_header = nrrd.read(options.in_lm) 
    else:
         lm = np.ones(np.shape(ct))   

    print "Reading patches segmentation..."
    in_patches, in_patches_header = nrrd.read(options.in_patches) 

    print "Compute histogram features..."
    kde_hist_extractor = kdeHistExtractor(lower_limit=np.int16(options.lower_limit), \
        upper_limit=np.int16(options.upper_limit))
    
    import timeit
    print(timeit.timeit("kde_hist_extractor.fit(ct, lm, in_patches)", setup="from kdeHistExtractor import fit"))
    kde_hist_extractor.fit(ct, lm, in_patches)
    
    
    if options.out_csv is not None:
        print "Writing..."
        kde_hist_extractor.df_.to_csv(options.out_csv, index=False)
        
