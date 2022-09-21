import SimpleITK as sitk
import argparse


def convert_data(input_image, output_image):
        img = sitk.ReadImage(input_image)
        sitk.WriteImage(img, output_image, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert nifti to nrrd')

    parser.add_argument("-i", dest="input_image", required=True, help='Input image name')
    parser.add_argument("-o", dest="output_image",required=False, help='Output image name.')

    op = parser.parse_args()

    in_image = op.input_image
    out_image = op.output_image

    convert_data(in_image, out_image)

