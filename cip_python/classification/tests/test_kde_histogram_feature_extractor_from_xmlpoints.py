import os.path
import pandas as pd
import nrrd

import numpy as np
import pdb
from pandas.util.testing import assert_frame_equal
import sys

#from cip_python.input_output.image_reader_writer import ImageReaderWriter
sys.path.append("/Users/rolaharmouche/ChestImagingPlatform/")
from cip_python.classification.kde_histogram_feature_extractor_from_xmlpoints \
  import kdeHistExtractorFromXML
from cip_python.input_output.image_reader_writer import ImageReaderWriter

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'
in_xml = this_dir + '/../../../Testing/Data/Input/simple_ct_regionAndTypePoints.xml'
ref_csv = this_dir + '/../../../Testing/Data/Input/simple_ct_histogramFeatures.csv'

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

