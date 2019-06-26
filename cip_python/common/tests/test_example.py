import os.path as osp
import tempfile
import SimpleITK as sitk
import numpy as np
import shutil

import cip_python.common as common

def test_example():
    """
    Test for demo purposes only.
    Simulate the output of an algorithm and compare to the basline image
    """
    # Make sure an input image is present
    input_file_path = common.Paths.testing_file_path("crop_ct_2slices.nrrd")
    assert osp.isfile(input_file_path), "Input image path file not found: {}".format(input_file_path)

    # Make sure the baseline image is present
    baseline_image_path = osp.join(osp.dirname(__file__), "baseline", "crop_ct_2slices.nrrd")
    assert osp.isfile(baseline_image_path), "Baseline file not found: {}".format(baseline_image_path)

    # Create a temp output path for the result
    temp_folder = tempfile.gettempdir()
    output_file_path = osp.join(temp_folder, "crop_ct_2slices.nrrd")

    # Simulate the execution of an algorithm (output => output_path)
    shutil.copy(baseline_image_path, output_file_path)

    # Compare the output to the baseline
    baseline_image = sitk.GetArrayFromImage(sitk.ReadImage(baseline_image_path))
    output_image = sitk.GetArrayFromImage(sitk.ReadImage(output_file_path))

    assert np.allclose(output_image, baseline_image), "The baseline image is different from the algorithm output"

