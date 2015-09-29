import os.path
import pandas as pd
import nrrd

import numpy as np
import pdb
from pandas.util.testing import assert_frame_equal
import sys

#from cip_python.input_output.image_reader_writer import ImageReaderWriter
sys.path.append("/Users/rolaharmouche/ChestImagingPlatform/")
from cip_python.classification.kde_histogram_feature_extractor \
  import kdeHistExtractor
  
np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'
ref_csv = this_dir + '/../../../Testing/Data/Input/simple_ct_histogramFeatures.csv'

def test_execute():
    ct_array, ct_header = nrrd.read(ct_name)

    grid_array = np.zeros(np.shape(ct_array)).astype(int)
    grid_array[:, :, 0] = 0
    grid_array[:, :, 1] = 1
    grid_array[:, :, 2] = 2

    lm_array = np.zeros(np.shape(ct_array)).astype(int)
    lm_array[:, :, 1] = 1
    
    hist_extractor = kdeHistExtractor(lower_limit=-1000, upper_limit=-30, \
        x_extent=5, y_extent = 11, z_extent = 1, in_patch_labels = grid_array,\
            lm= lm_array)
    hist_extractor.fit(ct_array, lm_array, grid_array)

    reference_df = pd.read_csv(ref_csv)
    assert_frame_equal(hist_extractor.df_, reference_df)


