import cip_python.common as common

def test_example_pythonCLI():
    """ Run a simple sample test"""
    ex = common.ExampleCLIClass()
    input_file = common.Paths.testing_file_path("crop_ct.nrrd")
    output_file = common.Paths.testing_output_file_path("crop_ct.smooth.nrrd")
    output = ex.execute(input_file, output_file)
    assert output is not None