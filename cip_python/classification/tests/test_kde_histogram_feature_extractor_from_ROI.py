import os.path
import pandas as pd
import nrrd
import numpy as np

from cip_python.classification import kdeHistExtractorFromROI
from cip_python.common import Paths

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

ct_name = Paths.testing_file_path('simple_ct.nrrd')
ref_csv = Paths.testing_file_path('simple_ct_histogramFeatures.csv')

def test_execute():
    ct_array, ct_header = nrrd.read(ct_name)
    ct_patch = np.squeeze(ct_array[:, :, 1])
    lm_patch = np.ones(np.shape(ct_patch))
    
    hist_extractor = kdeHistExtractorFromROI(lower_limit=-1000, upper_limit=-30)
    hist_extractor.fit(ct_patch, lm_patch)
    
    reference_df = pd.read_csv(ref_csv)
    reference_array = np.array(reference_df.filter(regex='hu'))[:,0:600]
    
    np.testing.assert_allclose(np.squeeze(np.transpose(reference_array)), hist_extractor.hist_[0:600],\
        err_msg='arrays not equal', verbose=True, rtol=1e-5)
    
