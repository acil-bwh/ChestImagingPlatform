import numpy as np

from cip_python.common import Paths
from cip_python.input_output.image_reader_writer import ImageReaderWriter
from cip_python.classification import Patcher

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

ct_name = Paths.testing_file_path('simple_ct.nrrd')

def test_execute():
    image_io = ImageReaderWriter()
    ct_array, ct_header = image_io.read_in_numpy(ct_name)

    extent = [5,11,1] 
    center = [2,5,1]
    patch_ref = np.squeeze(ct_array[:, :, 1])

    test_bounds = Patcher.get_bounds_from_center(ct_array,center,extent)
    test_patch = Patcher.get_patch_given_bounds(ct_array,test_bounds)
    
    np.testing.assert_array_equal(test_patch,patch_ref, err_msg='arrays not equal', verbose=True)

