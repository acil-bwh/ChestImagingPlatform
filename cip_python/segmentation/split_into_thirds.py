import argparse

import os
import numpy as np
import SimpleITK as sitk



from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter


class LungThirdSplitter():
    def __init__(self):
        self.size_th = 0.05
        self.coordinate_system = 'lps'
        c = ChestConventions()
        self.RightLabel = c.GetChestRegionValueFromName('RightLung')
        self.LeftLabel = c.GetChestRegionValueFromName('LeftLung')
        self.WholeLung = c.GetChestRegionValueFromName('WholeLung')
        self.UpperThird = c.GetChestRegionValueFromName('UpperThird')
        self.MiddleThrid = c.GetChestRegionValueFromName('MiddleThird')
        self.LowerThird = c.GetChestRegionValueFromName('LowerThird')
        self.LeftUpperThird = c.GetChestRegionValueFromName('LeftUpperThird')
        self.LeftMiddleThird = c.GetChestRegionValueFromName('LeftMiddleThird')
        self.LeftLowerThrid = c.GetChestRegionValueFromName('LeftLowerThird')
        self.RightUpperThird = c.GetChestRegionValueFromName('RightUpperThird')
        self.RightMiddleThrid = c.GetChestRegionValueFromName('RightMiddleThird')
        self.RightLowerThrid = c.GetChestRegionValueFromName('RightLowerThird')

        self.cc_f = sitk.ConnectedComponentImageFilter()
        self.r_f = sitk.RelabelComponentImageFilter()
        self.ls = sitk.LabelShapeStatisticsImageFilter()

    def execute(self, lm):
        # Get Region/Type Information
        lm_np = sitk.GetArrayFromImage(lm)
        lm_np = lm_np.astype(np.uint16)

        # Output holder copy
        olm_tmp = sitk.Image(lm)
        olm_np = sitk.GetArrayFromImage(olm_tmp)
        olm_np = olm_np.astype(np.uint16)

        present_labels = np.unique(lm_np)

        for ll in present_labels:
            lm_target_np = np.zeros(lm_np.shape, dtype=lm_np.dtype)
            lm_target_np[lm_np == ll] = 1

            tmp_itk = sitk.GetImageFromArray(lm_target_np)
            tmp_cc = self.cc_f.Execute(tmp_itk)
            tmp_rl = self.r_f.Execute(tmp_cc)

            tmp_rl_np = sitk.GetArrayFromImage(tmp_rl)
            olm_np[tmp_rl_np > 1] = 0

        lm_type_np = olm_np >> 8
        lm_region_np = olm_np & 255

        # Splitting in Thirds
        size = lm.GetSize()
        vol_right = np.sum(olm_np == self.RightLabel)
        vol_left = np.sum(olm_np == self.LeftLabel)
        target_vol_right = 0
        target_vol_left = 0
        for zz in range(size[2]):
            cut = olm_np[zz, :, :]
            right_mask = (cut == self.RightLabel)
            left_mask = (cut == self.LeftLabel)

            slice_vol_right = np.sum(right_mask)
            slice_vol_left = np.sum(left_mask)
            if target_vol_right <= vol_right / 3:
                cut[right_mask] = self.RightLowerThrid
            elif target_vol_right > vol_right / 3 and target_vol_right <= 2 * vol_right / 3:
                cut[right_mask] = self.RightMiddleThrid
            else:
                cut[right_mask] = self.RightUpperThird

            target_vol_right = target_vol_right + slice_vol_right

            if target_vol_left <= vol_left / 3:
                cut[left_mask] = self.LeftLowerThrid
            elif target_vol_left > vol_left / 3 and target_vol_left <= 2 * vol_left / 3:
                cut[left_mask] = self.LeftMiddleThird
            else:
                cut[left_mask] = self.LeftUpperThird

            target_vol_left = target_vol_left + slice_vol_left

        # Transfer type labels to output LM
        wl_mask = (lm_region_np == self.WholeLung) | (lm_region_np == self.RightLabel) | \
                  (lm_region_np == self.LeftLabel)

        olm_np[np.logical_not(wl_mask)] = lm_region_np[np.logical_not(wl_mask)]
        pp = olm_np + (lm_type_np << 8)

        olm = sitk.GetImageFromArray(pp)
        olm.CopyInformation(lm)
        return olm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a lung mask into thirds.')
    parser.add_argument('--i', dest='in_lm', help="Input labelmap file (.nrrd)", type=str, required=True)
    parser.add_argument('--o', dest='out_lm', help='Output labelmap file (.nrrd). If not declared, input labelmap is overwritten.', required=False, type=str,
                        default=None)


    op = parser.parse_args()

    print ('    Splitting Segmentation in Thirds...')

    lung_segmentation = sitk.ReadImage(op.in_lm)


    lung_splitter = LungThirdSplitter()
    lung_segmentation = lung_splitter.execute(lung_segmentation)

    if op.out_lm:

        sitk.WriteImage(lung_segmentation, op.out_lm, True)
    else:
    	sitk.WriteImage(lung_segmentation, op.in_lm, True)