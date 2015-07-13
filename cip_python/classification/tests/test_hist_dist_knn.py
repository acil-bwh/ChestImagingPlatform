from cip_python.classification.hist_dist_knn import HistDistKNN
import numpy as np
import pdb

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

def test_execute():
    N = 5
    bins = 10
    hists = np.random.randn(N, 10)
    hists[2::] += 100
        
    dists = np.random.rand(N)
    dists[2::] += 100

    y = np.array([1, 1, 2, 2, 2])

    test_hist = np.ones(bins)
    test_dist = 0.
    
    n_neighbors = 3
    beta = 10.
    clf = HistDistKNN(n_neighbors=n_neighbors, beta=beta)
    clf.fit(hists, dists, y)
    class_label = clf.predict(test_hist, test_dist)

    assert class_label == 1, "Class label not as expected"
    
