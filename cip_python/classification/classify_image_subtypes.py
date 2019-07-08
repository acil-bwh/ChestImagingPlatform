import numpy as np
from optparse import OptionParser
import time
import multiprocessing
from multiprocessing import Pool
import ctypes
try:
    import copy_reg
except:
    import copyreg as copy_reg
import types
import pdb
 
from scipy import ndimage
import pandas as pd

from cip_python.classification.localHistogramModel import LocalHistogramModel
from cip_python.classification.localHistogramMapper import LocalHistogramModelMapper

from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.classification import HistDistKNN
from cip_python.classification import kdeHistExtractorFromROI
from cip_python.classification import DistExtractorFromROI
from cip_python.classification import Patcher
from cip_python.segmentation.grid_segmenter import GridSegmenter
global ct, lm, distance_image
      
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)
          
class ParenchymaSubtypeClassifier:
    """General purpose class implementing a ct kernel density classification. 

    The user inputs CT numpy array and a mask. A patch array is generated and
    each patch where mask > 0 is segmented into one of the predefined classes.
    The output is a numpy array with a class label for each corresponding CT 
    voxel. 
       
    Parameters 
    ----------    
    
    lm: 3D numpy array, shape (L, M, N)
        Input lung mask for the CT that we want to classify
      
    distance_image: 3D numpy array, shape (L, M, N)
        Input distance image, where at each voxel, the distance to the lung is 
        recorded.
                                                
    kde_lower_limit:int
        Lower limit of kde histogram
        
    kde_upper_limit:int
        Upper limit of kde histogram

    n_neighbors: int
        Number of neighbours to use in the KNN 
        
    num_threads: int
        Number of threads for parallel processing
        
    n_patches_perbatch: int
        Number of patches to process at a given time (with a given thread)
        
    patch_size: list, shape (3,1)
        (x,y,z) Size of each patch to be classified
        
    feature_extent: list, shape (3,1)
        (x,y,z) Size of the ROI in which the features are to be extracted
        (centered at each patch)
    
    high_density_label: int
        type label to assign to high density patches (mean value > -250).
        Default 1=Normal Parenchyma
            
    Attribues
    ---------
    unique_patch_labels : N by 1 numpy array
        All the unique values assigned to patches, each
        of which requiring a classification. 
        
    patch_centers : N by 3 numpy array
        The center coordinates for each patch to be classified.

    knn_classifier : object
        The scikit learn KNN classifier
    """
    
    ct = None

    def __init__(self, lm = None, kde_lower_limit=-1024, 
            kde_upper_limit=0, beta=0.075, n_neighbors=5, num_threads = 1, 
            n_patches_perbatch = 10000, patch_size = [5,5,5], feature_extent = 
            [31,31,1], hist_comparison='l1_minkowski',high_density_label=1):
        
        self.kde_lower_limit = kde_lower_limit
        self.kde_upper_limit = kde_upper_limit

        self.beta = beta
        self.n_neighbors = n_neighbors
        self.n_patches_perbatch = n_patches_perbatch
        self.num_threads = num_threads

        self.patch_size = patch_size
        self.feature_extent = feature_extent
        
        self.knn_classifier = None
        self.hist_comparison = hist_comparison

        self.min_hu = -1024
        self.max_hu = - 1024+600
        
        self.high_density_label = high_density_label
        
        """ compute patches so that they can be used for obtaining centers"""
        mypatch_array = np.zeros(np.shape(lm)).astype(int)
        grid_segmentor = GridSegmenter(input_dimensions=None, ct=lm, \
                x_size=self.patch_size[0], y_size=self.patch_size[1],
                z_offset=self.patch_size[2])
        mypatch_array = grid_segmentor.execute()

        """ get all patch centers"""
        unique_patch_labels_temp = np.unique(mypatch_array[:])
        self.unique_patch_labels = \
            unique_patch_labels_temp[unique_patch_labels_temp>0]

        self.patch_centers = ndimage.measurements.center_of_mass(\
            mypatch_array, mypatch_array.astype(np.int32), \
            self.unique_patch_labels)        


    def fit(self, training_df):        
        """ train the classifier 
        
        Inputs
        ----------
        training_df: pandas dataframe
            contains all the features and associated classes for training data
            Only the first 600 entries of the HU hitogram are taken       
        """
        
        # import pdb
        # pdb.set_trace()
        # hu_cols=list()
        # for col_num in xrange(self.min_hu,self.max_hu):
        #     hu_cols.append('hu'+str(col_num))
        training_histograms = np.array(training_df.filter(regex='hu'))[:,0:600] 
        training_distances = np.squeeze(np.array(training_df.filter(\
            regex='Distance'))[:,:])


        training_classes = np.zeros(np.shape(training_df)[0], dtype = 'int')
        training_text_classes = training_df["ChestType"][:].values
        mychestConvention =ChestConventions()

        """ define a region type class number for each type value"""
        
        for i in range (0, np.shape(training_df)[0]):
            training_classes[i] = \
                mychestConvention.GetChestTypeValueFromName(\
                    training_text_classes[i])
                        
        """ train"""
        self.knn_classifier =  HistDistKNN(n_neighbors = self.n_neighbors, \
            beta = float(self.beta), hist_comparison=self.hist_comparison)
        self.knn_classifier.fit(training_histograms, training_distances, \
            training_classes)
        


    def process_patchset(self, patch_begin_index,high_density_label=1):
        """ processes a set of patches in parallel, having patch index starting  
        at patch_begin_index and ending at patch_begin_index+n_patches_perbatch
        
        Inputs
        ------
        patch_begin_index: int
            starting label of first patch to be processed
            
        Returns
        ----------
        
        output: dict()
            For each patch that was classified, this dict stored the patch
            bounds in the original CT as well as the classification      
        """
        
        patch_end_index = np.min([patch_begin_index+self.n_patches_perbatch,\
            np.shape(self.unique_patch_labels)[0]])

        predicted_values = [None]*(patch_end_index-patch_begin_index)
        patch_to_label_bounds = [None]*(patch_end_index-patch_begin_index)

        mychestConvention =ChestConventions()
        
        inc = 0
        global ct
        global lm
        global distance_image
        
        for patch_index in range(patch_begin_index, patch_end_index): 
            # unique_patch_labels_positive:
            """ Extract ROI """            
            """ classify and get patch bounds to store in output"""
            if (np.mod(patch_index,400) ==0):
                print("classifying patch"+str(patch_index))
                        
            patch_center = map(int, self.patch_centers[patch_index])
            patch_bounds = Patcher.get_bounds_from_center(ct,patch_center, self.feature_extent)
            patch_ct = Patcher.get_patch_given_bounds(ct,patch_bounds)
            patch_lm = Patcher.get_patch_given_bounds(lm,patch_bounds)
            patch_distance = Patcher.get_patch_given_bounds(distance_image,\
                patch_bounds)

            if (np.sum(patch_lm > 0) > 2):
                
                """ get  features from ROI"""
                my_hist_extractor = kdeHistExtractorFromROI(lower_limit=-1024,\
                    upper_limit=0)

                my_hist_extractor.fit( patch_ct, patch_lm)                   
            
                dist_extractor = DistExtractorFromROI(chest_region="WholeLung")
                dist_extractor.fit(patch_distance, patch_lm) 

                if (my_hist_extractor.hist_ is None):
                    raise ValueError("classify image subtypes:: Empty histogram")

                the_histogram = my_hist_extractor.hist_[0:600]
                #Rescaling distance to account for scale factor for short packing.
                the_distance = dist_extractor.dist_ / 100.0
                  
                predicted_value_tmp = self.knn_classifier.predict(the_histogram, the_distance)

                #Post kNN classification rule encoding
                #1. Set high density values to default label (ex. Normal parenchyma)
                if (np.median(patch_ct) > (-250)):
                    predicted_value_tmp = self.high_density_label
                                    
                predicted_values[inc] = \
                    [mychestConvention.GetValueFromChestRegionAndType(0, predicted_value_tmp)]

                patch_to_label_bounds[inc] = \
                    Patcher.get_bounds_from_center(ct,patch_center,  self.patch_size)
  
                inc += 1
        
        output = dict()
           
        predicted_values = predicted_values[0:inc]
        patch_to_label_bounds = patch_to_label_bounds[0:inc]
        
        output["predicted_values"] =  predicted_values
        output["patch_to_label_bounds"] =   patch_to_label_bounds                  
        return output
                
        
                                                                                                     
    def predict(self,  in_ct, in_lm, in_distance):  
        """ Perform the full classification
        
        Inputs
        ------
        training_df: pandas dataframe
            contains all  features and associated classes for the training data 
            
        Returns
        ----------
        
        ct_labels_array: 3D numpy array, shape (L, M, N)
            Output classification      
        """
        
        tic = time.clock()

        """
        Initialize parallel process and train the classifier
        """

        ct_labels_array = np.zeros(np.shape(in_ct))       
        num_patches_to_process = np.shape(self.unique_patch_labels)[0]        
 
        """
        Pack distance map in short array with two significant digits to improve memory efficiency
        """
        in_distance=(100.0*in_distance).astype('int16')

        """
        Make a shareable copy of the volumes 
        """
        global ct
        num_array_items = np.shape(in_ct)[0]* np.shape(in_ct)[1]* \
            np.shape(in_ct)[2]
        shared_array = multiprocessing.Array(ctypes.c_short, num_array_items, \
            lock=False)
        ct = np.frombuffer(shared_array, dtype = ctypes.c_short)
        ct = ct.reshape(np.shape(in_ct))
        ct[:] = in_ct[:]

        global lm
        num_array_items = np.shape(in_lm)[0]* np.shape(in_lm)[1]* \
            np.shape(in_lm)[2]
        shared_array_lm = multiprocessing.Array(ctypes.c_ushort, \
            num_array_items, lock=False)
        lm = np.frombuffer(shared_array_lm, dtype = ctypes.c_ushort)
        lm = lm.reshape(np.shape(in_lm))
        lm[:] = in_lm[:]

        global distance_image
        num_array_items = np.shape(in_distance)[0]* np.shape(in_distance)[1]* \
            np.shape(in_distance)[2]
        shared_array_distance = multiprocessing.Array(ctypes.c_short, \
            num_array_items, lock=False)

        distance_image = np.frombuffer(shared_array_distance, \
            dtype = ctypes.c_short)
        distance_image = distance_image.reshape(np.shape(in_distance))
        distance_image[:] = in_distance[:]

        """
        Run the classifier
        """        
        print("running classifier on "+str(num_patches_to_process)+" patches.")
        p = Pool(int(self.num_threads))
        parallel_result=(p.map(self.process_patchset, \
            range(0,num_patches_to_process,self.n_patches_perbatch)))

        #p.close()
        """
        Assign the output volume with the appropriate labels
        """                    
        for i in range(len(parallel_result)):
            for j in range(np.shape(parallel_result[i]["predicted_values"])[0]):                
                patch_processed_label=parallel_result[i]["predicted_values"][j]
                patch_processed_bounds = \
                    parallel_result[i]["patch_to_label_bounds"][j]                
                
                ct_labels_array[patch_processed_bounds[0]:patch_processed_bounds[1], \
                    patch_processed_bounds[2]:patch_processed_bounds[3],\
                    patch_processed_bounds[4]:patch_processed_bounds[5]] = \
                        patch_processed_label[0]
        
        ct_labels_array[lm<1] = 0   

        # mychestConvention = ChestConventions()
        # chest_type_val_not_imflamed = mychestConvention.GetChestTypeValueFromName('NORMALNOTINFLAMED')
        # ct_labels_array[ct_labels_array == 256] = [mychestConvention.GetValueFromChestRegionAndType(0, chest_type_val_not_imflamed)]

        toc = time.clock()
        print("execution time = "+str(toc - tic))                                                              
        return ct_labels_array
    
             
                
if __name__ == "__main__":
    desc = """ Classifies CT images into disease subtypes using kernel density
        estimation and k-nearest neighbours. It uses a feature vector 
        consisting of kde histogram bins and a distance from a certain 
        structure. The user must provide an input CT image, a distance map for 
        the distance feature, and a training file."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input nrrd CT file', dest='in_ct', 
                      metavar='<string>', default=None)
    parser.add_option('--in_lm',
                      help='Input nrrd mask file. The classification will only\
                      be computed in areas where the mask is > 0. If lm is not\
                      included, the histogram will be computed everywhere. And\
                      every where inside the ct will be considered lung', 
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('--in_distance',
                      help='Input distance nrrd file (distance to parenchymal \
                      wall). Generate one using ComputeDistanceMap command \
                      line tool.', 
                      dest='in_distance', metavar='<string>', default=None)                      
    parser.add_option('--out_label',
                      help='Output label file with the subtype \
                      classifications.', dest='out_label', \
                      metavar='<string>', default=None)                                         
    parser.add_option('--training_file',
                      help='File containing the feature vectors for the \
                        training data and associated labels. (HDF5)',
                      dest='in_training', metavar='<string>', action='append', default=None)
    parser.add_option('--n_neighbors',
                      help='Number of nearest neighbours.  (optional)',  
                      dest='n_neighbors', metavar='<string>', type=int, 
                      default=5)                            
    parser.add_option('--lower_limit',
                      help='kernel density estimation lower histogram limit.  \
                      (optional)',  dest='lower_limit', 
                      metavar='<string>', type=float, default=-1024)                        
    parser.add_option('--upper_limit',
                      help='kernel density estimation  upper histogram limit. \
                      (optional)',  dest='upper_limit', 
                      metavar='<string>', type=float, default=0)  
    parser.add_option('--dist_prop',
                      help='Proportion of the distance feature to be used in \
                      the knn  ', dest='dist_prop', metavar='<string>', 
                      type=float, default=0.075)         
    parser.add_option('--patch_size',
                      help='patch size in the x,y,z direction ', 
                      dest='patch_size', metavar='<string>', type=int, 
                      nargs=3, default=[10,10,10])
    parser.add_option('--feature_extent',
                      help=' region size in the x direction over which the \
                        features will be estimated. The region will be \
                        centered at the patch center.', dest='feature_extent', 
                        metavar='<string>', type=int, nargs=3, 
                        default=[31,31,1])
    parser.add_option('--n_patches_perbatch',
                      help='Number of patches to execute at a time. This is \
                      relevant when multithreading, the higher the number, \
                      the less paralel calls are done (optional)', 
                      dest='n_patches_perbatch', metavar='<string>', 
                      type=int, default=10000)
    parser.add_option('--high_density_label',
                      help='Label to use for patches whose mean value is above -250',
                      dest='high_density_label',metavar='<string>',
                      type=int,default=1)
    parser.add_option('--num_threads',
                      help='Number of threads in parallel.  (optional)',  
                      dest='num_threads', metavar='<string>', type=int, 
                      default=1)                                                                                                   
    parser.add_option('--hist_comparison',
                      help='distance metric for comparing histograms. \
                      (optional)',  dest='hist_comparison', 
                      metavar='<string>', default='l1_minkowski')

    parser.add_option('--config_file',
                      help='Json file with configuration to drop classes. \
                      (optional)',  dest='config_file',
                      metavar='<string>', default=None)

    (options, args) = parser.parse_args()                                                                   

    image_io = ImageReaderWriter()
    print ("Reading CT...")
    ct_array, ct_header = image_io.read_in_numpy(options.in_ct) 

    if (options.dist_prop >0):
        assert((options.in_distance is  not None) ), \
            " input distance map file must be provided when --dist_prop is set." 
    dist_array, distheader = image_io.read_in_numpy(options.in_distance) 
            
    lm_array = np.ones(np.shape(ct_array)).astype(int)   # ones by default
        
    if (options.in_lm is not None):
        print ("Reading mask...") 
        lm_array, lm_header = image_io.read_in_numpy(options.in_lm) 
         
       
    """
    Load training dataframe
    """ 
    print("Loading training data...")

    model_list=list()
    model_list = options.in_training

    training_model = LocalHistogramModel(model_list)

    if options.config_file is not None:
        mapper = LocalHistogramModelMapper()
        mapper.read_config_json(options.config_file)
        training_df = training_model.get_training_df_from_model()
        training_df = mapper.map_model(training_df)

    else:
        training_df = training_model.get_training_df_from_model()


    print("classifying ..") 
    my_classifier = ParenchymaSubtypeClassifier(lm=lm_array, \
        kde_lower_limit=options.lower_limit, \
        kde_upper_limit=options.upper_limit, beta=options.dist_prop, \
        n_neighbors=options.n_neighbors, num_threads = options.num_threads, \
        n_patches_perbatch = options.n_patches_perbatch, patch_size = \
        options.patch_size, feature_extent = options.feature_extent, \
        hist_comparison = options.hist_comparison,high_density_label=options.high_density_label)

    my_classifier.fit(training_df)
    ct_labels_array = my_classifier.predict(ct_array, lm_array, dist_array)

    assert(options.out_label is  not None), \
        "Output label file must be provided"
    
    print ("Writing output labels ")
    image_io.write_from_numpy(ct_labels_array,ct_header,options.out_label)                            
