import argparse
import numpy as np
from skimage.morphology import skeletonize
import SimpleITK as sitk

from cip_python.input_output.image_reader_writer import ImageReaderWriter


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method to reduce label map to 1 pixel wide representations')
    parser.add_argument('--i_lm', help="Path to labelmap (.nrrd)", type=str, required=True)
    parser.add_argument('--o', help="Path to save skeletonized labelmap (.nrrd)", type=str, required=True)
    parser.add_argument("--dilate", help='Optional. Set to dilate skeletonized label', action="store_true")
    parser.add_argument("--r", help='Radius for dilation', required=False, nargs='+', type=str,
                        default=[1, 1, 1])
    args = parser.parse_args()

    io = ImageReaderWriter()

    in_lm, metainfo = io.read_in_numpy(args.i_lm)
    in_lm[in_lm > 1] = 1
    out_lm = skeletonize(in_lm)

    if args.dilate:
        radius = [int(i) for i in args.r]

        out_lm_sitk = sitk.BinaryDilate(sitk.GetImageFromArray(out_lm), radius, kernelType=sitk.sitkCross)
        out_lm = sitk.GetArrayFromImage(out_lm_sitk)

    io.write_from_numpy(out_lm, metainfo, args.o)
