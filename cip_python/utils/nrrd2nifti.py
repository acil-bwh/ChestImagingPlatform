import SimpleITK as sitk
import argparse


def convert_nrrd2nifti(input_image, output_image):
        img = sitk.ReadImage(input_image)
        sitk.WriteImage(img, output_image, True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert nifti to nrrd')

    parser.add_argument("-i", dest="input_image", required=True, help='Input nrrd image name (.nrrd)')
    parser.add_argument("-o", dest="output_image",required=False, help='Optional. Output nifti image name. '
                                                                       'If not specified, the same name as input nrrd '
                                                                       'with new extension (.nii.gz) will be used')

    op = parser.parse_args()

    nrrd_image = op.input_image

    if op.output_image is None:
        nifti_image = op.input_image.split('/')[-1].split('.')[0] + ".nii.gz"
    else:
        nifti_image = op.output_image

    convert_nrrd2nifti(nrrd_image, nifti_image)

