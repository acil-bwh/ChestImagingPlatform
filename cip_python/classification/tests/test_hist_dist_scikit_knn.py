from cip_python.classification import HistDistScikitKNN
import numpy as np

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

def test_execute():
    N = 5
    bins = 10
    hists = np.random.randn(N, 10)
    hists[2::] += 100
        
    dists = np.random.rand(N)
    dists[2::] += 100

    y = np.array([1, 1, 4, 4, 4])

    test_hist = np.ones(bins)
    test_dist = 0.
    
    n_neighbors = 3
    beta = 10.
    clf = HistDistScikitKNN(n_neighbors=n_neighbors, beta=beta, \
        classes = [1,4,5])
    clf.fit(hists, dists, y)
    class_label = clf.predict(test_hist, test_dist)

    assert class_label == 1, "Class label not as expected"

    test_hists = 100 + np.ones([2, bins])
    test_dists = np.array([100., 100.])
    class_labels = clf.predict(test_hists, test_dists)
    assert class_labels[0] == 4 and class_labels[1] == 4, \
      "Class labels not as expected"
    
    #class_probs = clf.predict_proba(test_hists, test_dists)
          #assert(class_probs == np.array([[ 0., 1., 0.],
#[ 0., 1., 0.]])).all(), "Probabilities not as expected"


