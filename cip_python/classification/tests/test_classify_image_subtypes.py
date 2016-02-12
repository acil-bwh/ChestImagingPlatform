import os.path
import numpy as np
import sys
import pandas as pd
from pandas.util.testing import assert_frame_equal

from cip_python.input_output.image_reader_writer import ImageReaderWriter

from cip_python.classification.classify_image_subtypes \
  import ParenchymaSubtypeClassifier

    


  
np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
ct_name = this_dir + '/../../../Testing/Data/Input/ct-patch.nrrd'
labels_name = this_dir + '/../../../Testing/Data/Input/ct-patch_ILDClassification.nrrd'
dist_name = this_dir + '/../../../Testing/Data/Input/distance-patch.nrrd'
training_df_name = this_dir + '/../../../Resources/SubtypesClassifiation/subtype_training.csv'

def test_execute():
    image_io = ImageReaderWriter()
    ct_array, ct_header = image_io.read_in_numpy(ct_name)
    labels_ref_array, labels_heads = image_io.read_in_numpy(labels_name)
    dist_array, distheader = image_io.read_in_numpy(dist_name) 
    lm_array = np.ones(np.shape(ct_array)).astype(int)
    
    training_df = pd.read_csv(training_df_name)

    my_classifier = ParenchymaSubtypeClassifier(lm=lm_array, beta=0.075, patch_size = [5, 5, 1])
    
    my_classifier.fit(training_df)
    ct_labels_array = my_classifier.predict(ct_array, lm_array, dist_array)

    
    
    np.testing.assert_array_equal(ct_labels_array,labels_ref_array, err_msg='arrays not equal', verbose=True)


