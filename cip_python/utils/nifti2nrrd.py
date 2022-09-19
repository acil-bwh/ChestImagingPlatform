import SimpleITK as sitk
import argparse


def convert_nifti2nrrd(input_image, output_image):
        img = sitk.ReadImage(input_image)
        sitk.WriteImage(img, output_image, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert nifti to nrrd')

    parser.add_argument("-i", dest="input_image", required=True, help='Input nifti image name (.nii.gz)')
    parser.add_argument("-o", dest="output_image",required=False, help='Optional. Output nrrd image name. '
                                                                       'If not specified, the same name as input nifti '
                                                                       'with new extension (.nrrd) will be used')

    op = parser.parse_args()

    nifti_image = op.input_image

    if op.output_image is None:
        nrrd_image = op.input_image.split('/')[-1].split('.')[0] + ".nrrd"
    else:
        nrrd_image = op.output_image

    convert_nifti2nrrd(nifti_image, nrrd_image)

