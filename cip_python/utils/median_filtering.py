from __future__ import division
import SimpleITK as sitk
import sys
from argparse import ArgumentParser


if __name__ == '__main__':
    desc = "Perform median filtering of a volume"
    
    parser = ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_file', required=True, help='Input image volume file')
    parser.add_argument('-o', '--output_file', required=True, help='Output image volume file')
    parser.add_argument('-r', '--radius', required=False, nargs='+', type=int, help=' Median filter radius', default=[1,1,1] )
    
    args = parser.parse_args()
    in_im=sitk.ReadImage(args.input_file)
    mf=sitk.MedianImageFilter()
    if len(args.radius) != in_im.GetDimension():
      print('Image dimension and number of radii must be equal')
      sys.exit(1)
    out_im=mf.Execute(in_im,args.radius)
    sitk.WriteImage(out_im,args.output_file,True)


