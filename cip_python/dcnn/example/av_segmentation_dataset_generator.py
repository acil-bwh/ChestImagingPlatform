import argparse
import numpy as np
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy

from cip_python.input_output import ImageReaderWriter, H5DatasetStore, Axis
from cip_python.dcnn.data import DataProcessing
from cip_python.common import ChestConventions


class AVSegmentationDatasetGenerator:
    def __init__(self):
        self.sitk_in_out = ImageReaderWriter()
        self.data_processing = DataProcessing()

        self.resampling_spacing = [0.625, 0.625, 0.625]
        self.patch_size = [64, 64, 64]

    def execute(self, input_ct, input_lm, input_st, nb_patches, cid, sid, output_h5):
        generator = H5DatasetStore(output_h5)
        generator.create_h5_file(description='Dataset containing original images for AV classification with dcnn',
                                 key_names=('sid', 'cid'),
                                 keys_description="Element ids: subject id (sid) and case id (cid)",
                                 override_if_existing=True)

        in_ct_sitk = self.data_processing.resample_image_itk_by_spacing(self.sitk_in_out.read(input_ct),
                                                                        self.resampling_spacing,
                                                                        output_type=sitk.sitkInt16,
                                                                        interpolator=sitk.sitkBSpline)
        in_lm_sitk_res = self.data_processing.resample_image_itk_by_spacing(self.sitk_in_out.read(input_lm),
                                                                            self.resampling_spacing,
                                                                            output_type=sitk.sitkUInt16,
                                                                            interpolator=sitk.sitkLinear)
        in_st_sitk = self.data_processing.resample_image_itk_by_spacing(self.sitk_in_out.read(input_st),
                                                                        self.resampling_spacing,
                                                                        output_type=sitk.sitkUInt16,
                                                                        interpolator=sitk.sitkLinear)

        in_lm_np = self.sitk_in_out.sitkImage_to_numpy(in_lm_sitk_res)
        in_lm_np[in_lm_np > 1] = 1

        in_st_np = self.sitk_in_out.sitkImage_to_numpy(in_st_sitk)
        in_st_np[np.logical_and(in_st_np != 12800, in_st_np != 13056)] = 0
        in_st_np[in_st_np == 12800] = 1
        in_st_np[in_st_np == 13056] = 2
        in_st_sitk = self.sitk_in_out.numpy_to_sitkImage(in_st_np, sitk_image_template=in_lm_sitk_res)

        axes = [Axis.new_index(nb_patches),
                Axis("x_dim", "X image size", 64, Axis.TYPE_SPATIAL, Axis.UNIT_PIXEL),
                Axis("y_dim", "Y image size", 64, Axis.TYPE_SPATIAL, Axis.UNIT_PIXEL),
                Axis("z_dim", "Z image size", 64, Axis.TYPE_SPATIAL, Axis.UNIT_PIXEL)]

        generator.create_ndarray(name='CT_patches',
                                 general_description='Patches randomly extracted from the lung region '
                                                     'of the original CT images',
                                 axes=axes, dtype=np.short)

        generator.create_ndarray(name='AV_patches',
                                 general_description='Patches containing the AV classification label '
                                                     '(0=background, 1=artery, 2=vein)'.format(type),
                                 axes=axes, dtype=np.uint8)

        lung_mask = np.argwhere(in_lm_np[:-64, :-64, :-64] == 1)
        for nn in range(nb_patches):
            roi_index = lung_mask[np.random.randint(0, lung_mask.shape[0])].tolist()
            ct_crop = sitk.RegionOfInterest(in_ct_sitk, self.patch_size, roi_index)
            st_crop = sitk.RegionOfInterest(in_st_sitk, self.patch_size, roi_index)

            generator.insert_ndarray_single_point(ds_name='CT_patches',
                                                  data_array=self.sitk_in_out.sitkImage_to_numpy(ct_crop),
                                                  key_array=([sid, cid]),
                                                  spacing_array=ct_crop.GetSpacing(),
                                                  origin_array=ct_crop.GetOrigin(),
                                                  missing=0)
            generator.insert_ndarray_single_point(ds_name='AV_patches',
                                                  data_array=self.sitk_in_out.sitkImage_to_numpy(st_crop),
                                                  key_array=([sid, cid]),
                                                  spacing_array=st_crop.GetSpacing(),
                                                  origin_array=st_crop.GetOrigin(),
                                                  missing=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Low dose to high dose single h5 dataset creation')
    parser.add_argument('--ict', help='Input CT (.nrrd)', dest='in_ct', type=str, default=None)
    parser.add_argument('--ilm', help='Input partial lung label map file (.nrrd)', dest='in_lm', type=str, default=None)
    parser.add_argument('--ist', help='Input stenciled AV file (.nrrd)', dest='in_st', type=str, default=None)
    parser.add_argument('--nbp', help='Number of 64x64x64 patches to extract from original image', dest='nb_p',
                        type=int, default=None)
    parser.add_argument('--cid', help='Input case ID of the CT', dest='cid', type=str, default=None)
    parser.add_argument('--sid', dest='sid', help='Input subject ID of the CT', required=True, type=str, default=None)
    parser.add_argument('--o', help='Output filename (.h5)', dest='out_filename', required=True, default=None)
    op = parser.parse_args()

    gg = AVSegmentationDatasetGenerator()
    gg.execute(op.in_ct, op.in_lm, op.in_st, int(op.nb_p), op.cid, op.sid, op.out_filename)
