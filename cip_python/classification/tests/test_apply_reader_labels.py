import os.path
import pandas as pd
from cip_python.input_output.image_reader_writer import ImageReaderWriter
from cip_python.classification import LabelsReader
import numpy as np

from cip_python.common import Paths

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 
seg_file = Paths.testing_file_path('simple_roiSegmentation.nrrd')
plocs_file = Paths.testing_file_path('simple_regionAndTypePoints.csv')

def test_execute():
    image_io = ImageReaderWriter()
    seg, seg_header = image_io.read_in_numpy(seg_file)

    plocs_df = pd.read_csv(plocs_file)    

    features_df = \
      pd.DataFrame(columns=['patch_label', 'ChestRegion', 'ChestType'])
    features_df.loc[0] = [1, 'UndefinedRegion', 'UndefinedType']
    features_df.loc[1] = [2, 'UndefinedRegion', 'UndefinedType']

    reader = LabelsReader()
    reader.apply_reader_labels(seg, seg_header, features_df, plocs_df, None)

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
      


ct_name = Paths.testing_file_path('simple_ct.nrrd')
lm_name = Paths.testing_file_path('simple_lm.nrrd')