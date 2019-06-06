import cip_python.common as common
from cip_python.input_output import ImageReaderWriter

an = common.AnatomicStructuresManager()
start, size_ = an.get_structure_coordinates("/Users/jonieva/tmp/17601P_EXP_STD_JHU_COPD_dcnnStructuresDetection.xml",
                             common.ChestRegion.TRACHEACARINA, common.Plane.AXIAL)

reader = ImageReaderWriter()
sitk_im = reader.read("/Users/jonieva/tmp/17601P_EXP_STD_JHU_COPD.nrrd")


im2 = sitk_im[:, :, start[2]-4:start[2] + 4]
arr=reader.sitkImage_to_numpy(im2)

