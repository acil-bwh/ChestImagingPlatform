import SimpleITK as sitk
from argparse import ArgumentParser

from cip_python.input_output import ImageReaderWriter


def getBodyMask(ct_arr):
    minmax = sitk.MinimumMaximumImageFilter()
    minmax.Execute(ct_arr)
    max_value = minmax.GetMaximum()
    bd_msk = sitk.BinaryThreshold(ct_arr, lowerThreshold=-250, upperThreshold=max_value)

    bd_msk = sitk.BinaryErode(bd_msk, kernelType=sitk.sitkCross, kernelRadius=(3, 3, 3), boundaryToForeground=False)
    for ii in range(3):
        bd_msk = sitk.BinaryDilate(bd_msk, kernelType=sitk.sitkCross, kernelRadius=(3, 3, 3), boundaryToForeground=False)

    bd_msk[:, :, 0] = sitk.BinaryFillhole(bd_msk[:, :, 0])
    bd_msk[:, :, -1] = sitk.BinaryFillhole(bd_msk[:, :, -1])
    bd_msk = sitk.BinaryFillhole(bd_msk)

    for jj in range(2):
        bd_msk = sitk.BinaryErode(bd_msk, kernelType=sitk.sitkCross, kernelRadius=(3, 3, 0), boundaryToForeground=False)

    bd_msk = sitk.RelabelComponent(sitk.ConnectedComponent(bd_msk), sortByObjectSize=True)
    bd_msk = bd_msk == 1

    return bd_msk

if __name__ == '__main__':
    desc = "Extract body mask from chest CT"

    parser = ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_file', required=True, help='Input image volume file (.nrrd)')
    parser.add_argument('-o', '--output_file', required=True, help='Output body label map (.nrrd)')

    args = parser.parse_args()
    image_io = ImageReaderWriter()
    ct = image_io.read(args.input_file)
    out_img = getBodyMask(ct)

    image_io.write(out_img, args.output_file)

