import numpy as np
from numpy import sum, sort, abs
from collections import Counter
from sklearn.cross_validation import LeaveOneOut
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KNeighborsClassifier

import pdb

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
    def __init__(self, n_neighbors=5, beta=0, hist_comparison='manhattan', classes = None):
        self.n_neighbors_ = n_neighbors
        self.beta_ = beta
        self.hist_comparison_ = hist_comparison
        if classes is None:
            self.classes_ = classes
        else:    
            self.classes_ = np.unique(classes)
            
        self.scikit_knn = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm='ball_tree', \
            metric=hist_comparison)    

        
    def histdistmetric(x,y,hist_comparison, beta):
        """ assumes that the data is in the form : [histarray0, ..., histarrayN, distval]
        """
        x_hist = x[0:-1]
        x_dist = x[-1]
        y_hist = y[0:-1]
        y_dist = y[-1]  
    
    
        hist_comp = None
        if hist_comparison == 'l1_minkowski':
            hist_comp = sum(abs(x_hist - y_hist), 1)                    
        elif hist_comparison == 'euclidean':
            hist_comp = np.sqrt(sum(np.square(x_hist - y_hist), 1))             
        else:
            raise ValueError('Unsupported histogram comparison method')
    
        # Compute the distance component (dist_comp) of the overall metric
        # value
        dist_comp = abs(x_dist - y_dist)
    
        # Now compute the complete metric value
        metric_val = hist_comp + beta*dist_comp

        return metric_val
        
    
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
        
        """ If classes was not input into the knn. It could be input
            if we want to extract class probabilities for some classes
            that are not in the training data. """
        if (self.classes_ is None):
            self.classes_ = np.unique(y)
            
        """ Generate N by B+1 array to feed into scikit learn classifier. """     
        training_data = np.zeros([np.shape(hists)[0], np.shape(hists)[1]+1])
        training_data[:,0:np.shape(hists)[1]] = hists
        training_data[:,np.shape(hists)[1]] = dists
        
        self.uniqueclasses_ = np.unique(y)
                
        self.scikit_knn.fit(training_data, y)


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
            classification_data = np.zeros([hist.shape[0]+1])
        else:
            mult_samples = True
            n_samples = hist.shape[0]
            classification_data = np.zeros([n_samples, hist.shape[1]+1])

        """ Generate N by B+1 array to feed into scikit learn classifier. """     

        if mult_samples:
            assert hist.shape[0] == dist.shape[0], \
              "Mismatch between histogram and distance data dimension"
            class_label = np.zeros(n_samples)
            classification_data[:,0:hist.shape[1]] = hist
            classification_data[:,hist.shape[1]] = dist              

        else:
            classification_data[0:hist.shape[0]] = hist
            classification_data[hist.shape[0]] = dist               
        
               
        class_label = self.scikit_knn.predict(classification_data)
        
        return class_label
        

    def predict_proba(self, hist, dist):
        """Get the class probabilities for the input data point X.

        Parameters
        ----------
        hist : array, shape ( B ) or shape ( M, B )
            Histogram of the test sample or histograms of 'M' test samples.

        dist : float or array, shape ( M ) 
            Physical distance of test sample to structure of interest or 
            vector of 'M' distances for 'M' test samples.
        
        Returns
        -------
        class_probabilities : array, shape (M, n_classes )
            Array with class probabilities of the 'K' nearest neighbors 
            for the 'M' test samples. n_classes is the number of self.y_ 
            unique values. The classes are ordered in ascending order 
            as per the numpy unique command
        """
        
        if len(hist.shape) == 1:
            mult_samples = False
            n_samples = 1
            classification_data = np.zeros([hist.shape[0]+1])
        else:
            mult_samples = True
            n_samples = hist.shape[0]
            classification_data = np.zeros([n_samples, hist.shape[1]+1])

        """ Generate N by B+1 array to feed into scikit learn classifier. """     
        num_classes = self.classes_.shape[0] #2#np.shape(self.classes_)[0]
        
        if mult_samples:
            assert hist.shape[0] == dist.shape[0], \
              "Mismatch between histogram and distance data dimension"
            class_probabilities = np.zeros([n_samples, num_classes])
            classification_data[:,0:hist.shape[1]] = hist
            classification_data[:,hist.shape[1]] = dist              

        else:
            classification_data[0:hist.shape[0]] = hist
            classification_data[hist.shape[0]] = dist               
            class_probabilities = np.zeros([num_classes])
        
        # dictlist = [dict() for x in range(n)]
                
        class_probabilities_temp = self.scikit_knn.predict_proba(classification_data)
        
        """ reconsile between classes available in training data and classes required for probabilities"""
        if(self.uniqueclasses_ == self.classes_):
            class_probabilities = class_probabilities_temp
        else:
            the_class = 0
            for the_unique_class in range(0, np.shape(self.uniqueclasses_)[0]):
                print("the class "+str(the_class)+ " unique class"+str(the_unique_class))
                if self.classes_[the_class] == self.uniqueclasses_[the_unique_class]:
                    print("storing prob")
                    """ find relationship between location in unique and overall. Assuming both in ascending order"""
                    if mult_samples:
                        class_probabilities[:,the_class] = class_probabilities_temp[:,the_unique_class] 
                    else:
                        class_probabilities[the_class] = class_probabilities_temp[the_unique_class] 
                          
                    the_class = the_class+1  
                else:
                    if mult_samples:
                        class_probabilities[:,the_class] = 0
                    else:
                        class_probabilities[the_class] = 0          
        return class_probabilities