import numpy as np
from numpy import sum, sort, abs
from collections import Counter

import pdb

class HistDistKNN():
    """K-nearest neighbors using histogram and distance (HistDist) metric. The
    metric is given by: histogram_metric + beta*distance_metric, where 
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
    """
    def __init__(self, n_neighbors, beta, hist_comparison='l1_minkowski'):
        self.n_neighbors_ = n_neighbors
        self.beta_ = beta
        self.hist_comparison_ = hist_comparison
        
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

    def predict(self, hist, dist):
        """Predict the class label of the input data point.

        Parameters
        ----------
        hist : array, shape ( B )
            Histogram of the test sample

        dist : float
            Physical distance of test sample to structure of interest.
        
        Returns
        -------
        class_label : integer
            The class label of the most common of the 'K' nearest neighbors
        """
        # Compute the histogram component (hist_comp) of the overall metric
        # value
        hist_comp = None
        if self.hist_comparison_ == 'l1_minkowski':
            hist_comp = sum(abs(hist - self.hists_), 1)
        else:
            raise ValueError('Unsupported histogram comparison method')

        # Compute the distance component (dist_comp) of the overall metric
        # value
        dist_comp = abs(dist - self.dists_)

        # Now compute the complete metric values, and identify the nearest
        # neighbors
        metric_vals = hist_comp + self.beta_*dist_comp
        nth_metric_val = sort(metric_vals)[self.n_neighbors_-1]
        ids = metric_vals <= nth_metric_val

        class_label = Counter(self.y_[ids]).most_common(1)[0][0]

        return class_label
