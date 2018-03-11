import os
import tempfile
import SimpleITK as sitk

import cip_python.common as common

def test_example_pythonCLI():
    """ Run a simple sample test and compare to a baseline"""
    ex = common.ExampleCLIClass()
    input_file = common.Paths.testing_file_path("crop_ct.nrrd")
    #output_file = common.Paths.testing_baseline_file_path("crop_ct_smooth.nrrd")
    temp_folder = tempfile.gettempdir()
    output_file = os.path.join(temp_folder, "crop_ct_smooth.nrrd")
    output = ex.execute(input_file, output_file)
    # Compare output to the baseline
    a1 = sitk.GetArrayFromImage(output)
    baseline_image = common.Paths.testing_baseline_file_path("crop_ct_smooth.nrrd")
    a2 = sitk.GetArrayFromImage(sitk.ReadImage(baseline_image))
    # Compare the two images
    assert (a1 - a2).sum() == 0

