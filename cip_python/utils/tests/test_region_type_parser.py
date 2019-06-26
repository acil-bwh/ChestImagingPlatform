import os.path
import numpy as np
import unittest

from cip_python.utils import RegionTypeParser
from cip_python.input_output import ImageReaderWriter

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 


class BaseTestMethods(unittest.TestCase):
    __test__=False
    def test_label_equality(self):
        assert self.parser.labels_.size == self.gt_labels.size, \
            "Number of labels not as expected"
      
        for gt_label in self.gt_labels:
            labels_match = False
            for label in self.parser.labels_:
                if label == gt_label:
                    labels_match = True
            assert labels_match, "Labels do not match with ground truth"


    def test_get_chest_regions(self):
    
        assert np.sum(np.sort(self.gt_regions) == np.sort(self.parser.get_chest_regions())) \
            == len(self.gt_regions), "Retrieved chest regions not as expected"

    def test_get_all_chest_regions(self):

      assert np.sum(np.sort(self.gt_all_regions) == \
                    np.sort(self.parser.get_all_chest_regions())) == len(self.gt_all_regions), \
        "Retrieved chest regions not as expected"

    def test_get_chest_types(self):
    
        assert np.sum(np.sort(self.gt_types) == np.sort(self.parser.get_chest_types())) \
            == len(self.gt_types), "Retrieved chest types not as expected"
    
    def test_get_all_pairs(self):

        assert np.sum(self.parser.get_all_pairs() == self.gt_all_pairs) == len(self.gt_all_pairs.flatten()), \
            "Retrieved pairs are not as expected"


class SimpleTest(BaseTestMethods):
    __test__=True
    def setUp(self):
        this_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = this_dir + '/../../../Testing/Data/Input/simple_lm.nrrd'
        image_io = ImageReaderWriter()
        im, header = image_io.read_in_numpy(file_name) 
        self.parser = RegionTypeParser(im)

        self.gt_labels = np.array([0, 2, 3, 512, 514, 515, 770, 771])
        self.gt_regions = np.array([0, 2, 3], dtype=int)
        self.gt_all_regions = np.array([0, 1, 2, 3], dtype=int)
        self.gt_types = np.array([0, 2, 3], dtype=int)
        self.gt_all_pairs = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [0, 2], [1, 2],
                               [2, 2], [3, 2], [1, 3], [2, 3], [3, 3]])


    def test_get_mask(self):
        X = 5
        Y = 11
        Z = 3
        gt_mask = np.empty([5, 11, 3], dtype=bool)
        
        # Test if we can correctly extract the WHOLELUNG region
        gt_mask[:, :, :] = False
        gt_mask[1:4, 1:4, 1] = True
        gt_mask[1:4, 7:10, 1] = True
        
        assert np.sum(self.parser.get_mask(chest_region=1) == gt_mask) == X*Y*Z, \
            "Mask is not as expected"
    
        # Test if we can correctly extract the AIRWAY type
        gt_mask[:, :, :] = False
        gt_mask[2, 5, 0] = True
        gt_mask[2, 5, 1] = True
        gt_mask[2, 2, 1] = True
        gt_mask[2, 8, 1] = True
        
        assert np.sum(self.parser.get_mask(chest_type=2) == gt_mask) == X*Y*Z, \
            "Mask is not as expected"
        
        # Test if we can correctly extract LEFTLUNG, AIRWAY
        gt_mask[:, :, :] = False
        gt_mask[2, 8, 1] = True
        
        assert np.sum(self.parser.get_mask(chest_region=3, chest_type=2) == gt_mask) == \
            X*Y*Z, "Mask is not as expected"

class ArrayTest(BaseTestMethods):
    __test__=True
    def setUp(self):
        lm=np.zeros([10,10,10],dtype=np.uint16)
        lm[:,1,:]=9
        lm[:,2,:]=10
        lm[:,3,:]=11
        lm[:,5,:]=12
        lm[:,6,:]=13
        lm[:,7,:]=14

        self.parser = RegionTypeParser(lm)

        self.gt_labels = np.array([0,9,10,11,12,13,14], dtype=int)
        self.gt_regions = np.array([0,9,10,11,12,13,14], dtype=int)
        self.gt_all_regions = np.array([0,1,2,3,9,10,11,12,13,14,20,21,22], dtype=int)
        self.gt_types = np.array([0], dtype=int)
        self.gt_all_pairs = np.array([[0, 0], [1, 0], [3, 0], [9, 0], [20,0], [10, 0],
                               [21,0], [11, 0], [22,0], [2, 0],
                              [12, 0], [13, 0], [14, 0]])
