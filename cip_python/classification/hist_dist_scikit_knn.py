import numpy as np
from sklearn.neighbors import NearestNeighbors
from cip_python.classification import HistDistKNN

class HistDistScikitKNN():
    """K-nearest neighbors using histogram and distance a user defined metric. Uses 
    Scikit learnn knn. The metric is given by: histogram_metric + beta*distance_metric, where 
    'histogram_metric' as any valid histogram metric (see below for supported 
    metrics), 'distance_metric' is the Euclidean distance between two distance 
    values, and 'beta' is a weight controlling the trade-off between the two
    components.

    Parameters
    ----------
    n_neighbors : integer
        The number of nearest neighbors to consider when classifying a new data
        point

    beta : float
        The weight controlling the trade-off between the distance term and the
        histogram term in the metric computation.

    hist_comparison : string, optional
        Indicates what histogram comparison method to use when computing the
        histogram term in the metric.    
    
    classes : numpy array, shape (num_classes, 1)
        contains the unique class labels. If None, will be inferred from the
        training data.
        
    """
    def __init__(self, n_neighbors=5, n_distance_samples = 500, beta=0, hist_comparison='l1_minkowski', classes = None):
        self.n_neighbors_ = n_neighbors
        self.beta_ = beta
        self.hist_comparison_ = hist_comparison
        if classes is None:
            self.classes_ = classes
        else:    
            self.classes_ = np.unique(classes)
            
        self.scikit_metric_ = None    
        if hist_comparison == 'l1_minkowski':
            self.scikit_metric_ = 'manhattan'             
        elif hist_comparison  == 'euclidean':
            self.scikit_metric_ = 'euclidean'   
  
        self.n_distance_samples_  = n_distance_samples
        self.scikit_knn = None

    def fit(self, hists, dists, y):
        """Supply the training data to the classifier.

        Parameters
        ----------
        hists : array , shape ( N, B )
            Each of the 'N' rows is a histogram with 'B' bins. Each row 
            corresponds to one sample in the training set.

        dists : array , shape ( N )
            Each of the 'N' values represents the physical distance of the
            sample to a structure of interest (such as the chest wall)

        y : array , shape ( N )
            Each of the 'N' values indicates the class label of the data             
        """
        assert hists.shape[0] == dists.shape[0] == y.shape[0], \
          "Data shape mismatch"        
        
        self.hists_ = hists
        self.dists_ = dists
        self.y_ = y
        
        #import pdb
        #pdb.set_trace()
        n_dist_samples = min(self.n_distance_samples_, hists.shape[0])
        self.scikit_knn = NearestNeighbors(n_dist_samples, algorithm='ball_tree', metric=self.scikit_metric_)  


        """ If classes was not input into the knn. It could be input
            if we want to extract class probabilities for some classes
            that are not in the training data. """
                                
        if (self.classes_ is None):
            self.classes_ = np.unique(y)
            
        #""" Generate N by B+1 array to feed into scikit learn classifier. """     
        #training_data = np.zeros([np.shape(hists)[0], np.shape(hists)[1]+1])
        #training_data[:,0:np.shape(hists)[1]] = hists
        #training_data[:,np.shape(hists)[1]] = dists
        
        """ Feed only intensity data into classifier """
        self.uniqueclasses_ = np.unique(y)
                
        self.scikit_knn.fit(hists, y)


    def predict(self, hist, dist):
        """Predict the class label of the input data point.

        Parameters
        ----------
        hist : array, shape ( B ) or shape ( M, B )
            Histogram of the test sample or histograms of 'M' test samples.

        dist : float or array, shape ( M ) 
            Physical distance of test sample to structure of interest or 
            vector of 'M' distances for 'M' test samples.
        
        Returns
        -------
        class_label : integer or vector array, shape ( M )
            The class label of the most common of the 'K' nearest neighbors or
            a vector containing class labels for the 'M' test samples (if 
            multiple test samples are specified).
        """
        if len(hist.shape) == 1:
            mult_samples = False
            n_samples = 1
            #classification_data = np.zeros([n_samples, hist.shape[0]+1])
        else:
            mult_samples = True
            n_samples = hist.shape[0]
            #classification_data = np.zeros([n_samples, hist.shape[1]+1])

        """ Generate N by B+1 array to feed into scikit learn classifier. """     

        if mult_samples:
            assert hist.shape[0] == dist.shape[0], \
              "Mismatch between histogram and distance data dimension"
            class_label = np.zeros(n_samples)
            #classification_data[:,0:hist.shape[1]] = hist
            #classification_data[:,hist.shape[1]] = dist              

        #else:
        #    classification_data[0,0:hist.shape[0]] = hist
        #    classification_data[0,hist.shape[0]] = dist               
        

        #print(hist.shape)
        #print(np.shape(hist.shape))
        
        if (np.shape(hist.shape)[0] > 1):

            subset_indeces = self.scikit_knn.kneighbors(hist, return_distance=False)[0]
        else:
            subset_indeces = self.scikit_knn.kneighbors([hist], return_distance=False)[0]
            
        subset_training_histogram = self.hists_[subset_indeces]
        subset_training_distance = self.dists_[subset_indeces]
        subset_training_class = self.y_[subset_indeces]

        my_knn_classifier2 = HistDistKNN(n_neighbors = self.n_neighbors_ , beta = self.beta_) #HistDistKNN
        my_knn_classifier2.fit(subset_training_histogram, subset_training_distance, subset_training_class)
        class_label = my_knn_classifier2.predict(hist, dist)

        return class_label
        

