import os
import numpy as np
import pandas as pd

from cip_python.input_output import ImageReaderWriter
from cip_python.classification  import ParenchymaSubtypeClassifier
from cip_python.classification import LocalHistogramModel
from cip_python.common import Paths

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

ct_name = Paths.testing_file_path('ct-patch.nrrd')
labels_name = Paths.testing_file_path('ct-patch_ILDClassification.nrrd')
dist_name = Paths.testing_file_path('distance-patch.nrrd')
training_df_name = Paths.resources_file_path('SubtypesClassification/subtype_training.csv')
training_h5_name = Paths.resources_file_path('SubtypesClassification/subtype_training.h5')

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

def test_execute2():
    image_io = ImageReaderWriter()
    ct_array, ct_header = image_io.read_in_numpy(ct_name)
    labels_ref_array, labels_heads = image_io.read_in_numpy(labels_name)
    dist_array, distheader = image_io.read_in_numpy(dist_name)
    lm_array = np.ones(np.shape(ct_array)).astype(int)
    
    training_model = LocalHistogramModel([training_h5_name])
    training_df = training_model.get_training_df_from_model()

    my_classifier = ParenchymaSubtypeClassifier(lm=lm_array, beta=0.075, patch_size = [5, 5, 1])
    
    my_classifier.fit(training_df)
    ct_labels_array = my_classifier.predict(ct_array, lm_array, dist_array)

    np.testing.assert_array_equal(ct_labels_array,labels_ref_array, err_msg='arrays not equal', verbose=True)

