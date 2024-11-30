from cip_python.utils.remap_lm import *
from cip_python.input_output import ImageReaderWriter
from cip_python.common import Paths

np.set_printoptions(precision = 3, suppress = True, threshold=1e6,
                    linewidth=200) 

file_name = Paths.testing_file_path('simple_lm.nrrd')
image_io = ImageReaderWriter()
lm, header = image_io.read_in_numpy(file_name)

def test_map_regions():
    remapped_lm = remap_lm(lm, region_maps=[['LeftLung', 'WholeLung']])

    remapped_lm_gt = np.copy(lm)
    remapped_lm_gt[1:4, 7:10, 1] = np.array([[  1,   1,   1],
                                             [769, 513,   1],
                                             [  1,   1,   1]], dtype=np.uint16)
    assert np.sum(remapped_lm == remapped_lm_gt) == 165, \
        "Output not as expected"

def test_map_types():
    remapped_lm = remap_lm(lm, type_maps=[['Airway', 'UndefinedType']])

    remapped_lm_gt = np.copy(lm)
    remapped_lm_gt[2, 5, 0] = 0
    remapped_lm_gt[2, 5, 1] = 0    
    remapped_lm_gt[2, 2, 1] = 2
    remapped_lm_gt[2, 8, 1] = 3    

    assert np.sum(remapped_lm == remapped_lm_gt) == 165, \
        "Output not as expected"

def test_map_pairs():
    remapped_lm = remap_lm(lm, pair_maps=[[['LeftLung', 'Airway'], 
                                           ['RightLung', 'Vessel']]])

    remapped_lm_gt = np.copy(lm)
    remapped_lm_gt[2, 8, 1] = 770

    assert np.sum(remapped_lm == remapped_lm_gt) == 165, \
        "Output not as expected"
