import os.path
from cip_python.utils.lm_parser import *
import nrrd
import pdb

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
file_name = this_dir + '/../../../Testing/Data/Input/simple_lm.nrrd'
im, header = nrrd.read(file_name) 
parser = LMParser(im)

gt_labels = np.array([0, 2, 3, 512, 514, 515, 770, 771])

def test_label_equality():
    assert parser.labels_.size == gt_labels.size, \
        "Number of labels not as expected"
    
    for gt_label in gt_labels:
        labels_match = False
        for label in parser.labels_: 
            if label == gt_label:
                labels_match = True
        assert labels_match, "Labels do not match with ground truth"

def test_get_mask():
    X = 5
    Y = 11
    Z = 3
    gt_mask = np.empty([5, 11, 3], dtype=bool)

    # Test if we can correctly extract the WHOLELUNG region
    gt_mask[:, :, :] = False
    gt_mask[1:4, 1:4, 1] = True
    gt_mask[1:4, 7:10, 1] = True    

    assert np.sum(parser.get_mask(chest_region=1) == gt_mask) == X*Y*Z, \
        "Mask is not as expected"

    # Test if we can correctly extract the AIRWAY type
    gt_mask[:, :, :] = False
    gt_mask[2, 5, 0] = True
    gt_mask[2, 5, 1] = True    
    gt_mask[2, 2, 1] = True    
    gt_mask[2, 8, 1] = True

    assert np.sum(parser.get_mask(chest_type=2) == gt_mask) == X*Y*Z, \
        "Mask is not as expected"

    # Test if we can correctly extract LEFTLUNG, AIRWAY
    gt_mask[:, :, :] = False
    gt_mask[2, 8, 1] = True

    assert np.sum(parser.get_mask(chest_region=3, chest_type=2) == gt_mask) == \
        X*Y*Z, "Mask is not as expected"

def test_get_chest_regions():
    gt_regions = np.array([0, 2, 3], dtype=int)

    assert np.sum(np.sort(gt_regions) == np.sort(parser.get_chest_regions())) \
        == 3, "Retrieved chest regions not as expected"

def test_get_all_chest_regions():
    gt_regions = np.array([0, 1, 2, 3], dtype=int)

    assert np.sum(np.sort(gt_regions) == \
                  np.sort(parser.get_all_chest_regions())) == 4, \
                  "Retrieved chest regions not as expected"

def test_get_chest_types():
    gt_types = np.array([0, 2, 3], dtype=int)

    assert np.sum(np.sort(gt_types) == np.sort(parser.get_chest_types())) \
        == 3, "Retrieved chest types not as expected"

def test_get_all_pairs():
    gt_pairs = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 2], [1, 2],
                         [2, 2], [3, 2], [1, 3], [2, 3], [3, 3]])

    assert np.sum(parser.get_all_pairs() == gt_pairs) == 22, \
        "Retrieved pairs are not as expected"

