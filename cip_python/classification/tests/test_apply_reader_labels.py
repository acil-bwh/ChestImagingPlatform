import os.path
import pandas as pd
import nrrd
from cip_python.classification.apply_reader_labels \
  import apply_reader_labels
import numpy as np
import pdb
from pandas.util.testing import assert_frame_equal

#from cip_python.input_output.image_reader_writer import ImageReaderWriter

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
seg_file = this_dir + '/../../../Testing/Data/Input/simple_roiSegmentation.nrrd'
plocs_file = this_dir + '/../../../Testing/Data/Input/simple_regionAndTypePoints.csv'

def test_execute():
    seg, seg_header = nrrd.read(seg_file) 
    plocs_df = pd.read_csv(plocs_file)    

    features_df = \
      pd.DataFrame(columns=['patch_label', 'ChestRegion', 'ChestType'])
    features_df.loc[0] = [1, 'UndefinedRegion', 'UndefinedType']
    features_df.loc[1] = [2, 'UndefinedRegion', 'UndefinedType']
      
    apply_reader_labels(seg, seg_header, features_df, plocs_df, None)

    index = features_df['patch_label'] == 1
    assert features_df.ix[index, 'ChestRegion'].values[0] == 'RightLung', \
      "ChestRegion not as expected"
    assert features_df.ix[index, 'ChestType'].values[0] == 'Airway', \
      "ChestType not as expected"

    index = features_df['patch_label'] == 2
    assert features_df.ix[index, 'ChestRegion'].values[0] == 'LeftLung', \
      "ChestRegion not as expected"
    assert features_df.ix[index, 'ChestType'].values[0] == 'UndefinedType', \
      "ChestType not as expected"      
      
