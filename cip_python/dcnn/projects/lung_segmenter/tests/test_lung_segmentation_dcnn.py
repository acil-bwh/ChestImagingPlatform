import os
import os.path as osp
import tempfile
import SimpleITK as sitk
import numpy as np

import cip_python.common as common
from cip_python.dcnn.logic import DeepLearningModelsManager
from cip_python.dcnn.projects.lung_segmenter import LungSegmenterDCNN

def test_lung_segmentation_dcnn():
    """ Run a simple sample test and compare to a baseline"""
    input_file_path = common.Paths.testing_file_path("crop_ct_2slices.nrrd")
    baseline_image_path = osp.join(osp.dirname(__file__), "baseline", "crop_ct_2slices_dcnnLungSegmentationLabelmap.nrrd")
    assert osp.isfile(baseline_image_path), "Baseline file not found: {}".format(baseline_image_path)

    temp_folder = tempfile.mkdtemp()
    output_file_path = os.path.join(temp_folder, "crop_ct_2slices_dcnnLungSegmentationLabelmap.nrrd")

    segmenter = LungSegmenterDCNN()

    # Load the model
    model_manager = DeepLearningModelsManager()
    axial_model_path = model_manager.get_model('LUNG_SEGMENTATION_AXIAL')
    print("*** Model used: {}".format(axial_model_path))

    segmentation = segmenter.execute(input_file_path, axial_model_path, None, segmentation_type='axial',
                                     N_subsampling=1)
    sitk.WriteImage(segmentation, output_file_path)

    baseline_image = sitk.GetArrayFromImage(sitk.ReadImage(baseline_image_path))
    output_image = sitk.GetArrayFromImage(sitk.ReadImage(output_file_path))

    print("*** Baseline: {}\nOutput:{}".format(baseline_image_path, output_file_path))

    assert np.allclose(output_image, baseline_image), "The baseline image ({}) is different from the algorithm output ({})".format(
        baseline_image_path, output_file_path)

    print("Test Passed!")

if __name__ == "__main__":
    test_lung_segmentation_dcnn()