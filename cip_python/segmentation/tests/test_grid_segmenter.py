import os.path
import pandas as pd
#import nrrd
from cip_python.segmentation.grid_segmenter import GridSegmenter
import numpy as np
import pdb
from cip_python.input_output.image_reader_writer import ImageReaderWriter

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

this_dir = os.path.dirname(os.path.realpath(__file__))
ct_name = this_dir + '/../../../Testing/Data/Input/simple_ct.nrrd'
grid_name = this_dir + '/../../../Testing/Data/Input/simple_grid.nrrd'

def test_execute():
    image_io = ImageReaderWriter()
    ct_array,ct_header=image_io.read_in_numpy(ct_name)
    grid_array, grid_header = image_io.read_in_numpy(grid_name)
    
    grid_segmenter = GridSegmenter(input_dimensions=None, ct=ct_array, \
        x_size=3, y_size=5, z_offset=2)
    grid_segmentation = grid_segmenter.execute()
    assert np.sum(np.abs(grid_segmentation-grid_array))==0, 'grid not as \
        expected'





