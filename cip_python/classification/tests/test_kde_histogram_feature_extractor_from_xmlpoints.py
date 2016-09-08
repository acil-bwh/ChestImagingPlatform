import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal

from cip_python.classification import kdeHistExtractorFromXML
from cip_python.input_output import ImageReaderWriter
from cip_python.common import Paths

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

ct_name = Paths.testing_file_path('simple_ct.nrrd')
in_xml = Paths.testing_file_path('simple_ct_regionAndTypePoints.xml')
ref_csv = Paths.testing_file_path('simple_ct_histogramFeatures.csv')

def test_execute():
    
    image_io = ImageReaderWriter()
    ct_array, ct_header=image_io.read_in_numpy(ct_name)


    lm_array = np.zeros(np.shape(ct_array)).astype(int)
    lm_array[:, :, 1] = 1
    with open(in_xml, 'r+b') as f:
        xml_data = f.read()
    print "Compute histogram features..."
    hist_extractor = kdeHistExtractorFromXML(lower_limit=-1000, upper_limit=-30, \
        x_extent=5, y_extent = 11, z_extent = 1)
    hist_extractor.fit(ct_array, ct_header, lm_array, xml_data)   


    reference_df = pd.read_csv(ref_csv)

    assert_frame_equal(hist_extractor.df_, reference_df)

