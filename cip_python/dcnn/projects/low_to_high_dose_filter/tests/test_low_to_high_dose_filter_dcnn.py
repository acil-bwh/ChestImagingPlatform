import os
import tempfile

import cip_python.common as common
from cip_python.dcnn.logic import DeepLearningModelsManager
from cip_python.dcnn.projects.low_to_high_dose_filter import LowToHighDoseFilterDCNN


def test_low_to_high_dose_dcnn():
    """ Run a simple sample test and compare to a baseline"""
    input_file_path = common.Paths.testing_file_path("crop_ct.nrrd")

    temp_folder = tempfile.mkdtemp()
    output_file_path = os.path.join(temp_folder, "crop_ct_2slices_dcnnL2HDose.nrrd")

    filter = LowToHighDoseFilterDCNN()

    # Load the model
    model_manager = DeepLearningModelsManager()
    model_path = model_manager.get_model('LOW_DOSE_TO_HIGH_DOSE')
    print("*** Model used: {}".format(model_path))

    try:
        filter.execute(input_file_path, model_path, output_file_path)
    except:
        print ("An error has occured while executing the algorithm")

    print("Test Passed!")

if __name__ == "__main__":
    test_low_to_high_dose_dcnn()
