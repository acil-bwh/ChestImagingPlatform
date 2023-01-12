"""
File: comptue_strain_and_eigenmaps.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    This class compute the strain and eigenvalueas from a dense deformation
    field and save them according to the parameters used.
"""

import numpy as np
import SimpleITK as sitk
import itk
from cip_python.registration.tx2dfield import Transform2DenseField
from cip_python.utils.itk_sitk_support import sitkImageToITK
from cip_python.utils.itk_sitk_support import copyImageInformation


def numpyImageSSRTensorToITK(array, ref_img):
    imgType = itk.Image[itk.SymmetricSecondRankTensor[itk.F, 3], 3]
    img = itk.image_from_array(array, ttype=(imgType, ))
    copyImageInformation(ref_img, img)
    return img


def castImageSSRTensorToDouble(img):
    # NOTE: itk.cast_image_filter do not support Symmetric rank tensor images
    dim = img.GetImageDimension()
    imgType = itk.Image[itk.SymmetricSecondRankTensor[itk.D, dim], dim]
    array = itk.array_view_from_image(img).astype(np.double)
    cast_img = itk.image_from_array(array, ttype=(imgType, ))
    copyImageInformation(img, cast_img)
    return cast_img


class StrainFilter():
    """This class compute the inf,, eulerian or lagrangian strain from a dense
    deformation field"""
    INFINITESIMAL = 0
    LAGRANGIAN = 1
    EULERIAN = 2

    def __init__(self, strain_form=INFINITESIMAL):
        self.strain_form = strain_form
        self.dfield_type = itk.Image[itk.Vector[itk.F, 3], 3]

    def Execute(self, dfield):
        if isinstance(dfield, sitk.Image):
            dfield = sitkImageToITK(dfield)
        elif not isinstance(dfield, itk.Image) and not isinstance(
                dfield, itk.VectorImage):
            raise Exception("Unknown image type (Supported: ITK and SITK)")

        # Cast dense field to dfield_ype, i.e itk.Image[itk.Vector[itk.F,3], 3]
        dfield = itk.cast_image_filter(dfield,
                                       ttype=(type(dfield), self.dfield_type))
        strain_filter = itk.StrainImageFilter[type(dfield), itk.F,
                                              itk.F].New(dfield)

        if self.strain_form == self.INFINITESIMAL:
            strain_filter.SetStrainForm(
                strain_filter.StrainFormType_INFINITESIMAL)
        elif self.strain_form == self.EULERIAN:
            strain_filter.SetStrainForm(
                strain_filter.StrainFormType_EULERIANALMANSI)
        elif self.strain_form == self.LAGRANGIAN:
            strain_filter.SetStrainForm(
                strain_filter.StrainFormType_GREENLAGRANGIAN)
        else:
            raise Exception(
                "Strain from should be inf. eulerian or lagrangian")

        strain_filter.Update()
        strain = strain_filter.GetOutput()
        # Image Type: symmetric second rank tensor SSRT
        return strain


class ComputeEigenValues():

    def __init__(self, order_by):
        self.order_by = order_by

    def Execute(self, strain):
        # SymmetricEigenFilter requires double not float
        if isinstance(strain, itk.itkImagePython.itkImageSSRTF33):
            strain = castImageSSRTensorToDouble(strain)

        eigen_img = itk.symmetric_eigen_analysis_image_filter(
            strain,
            order_eigen_values_by=self.order_by,
            dimension=strain.GetImageDimension())

        eigen_array = itk.array_view_from_image(eigen_img)
        dim = strain.GetImageDimension()
        imgs = []
        for i in range(dim):
            eigen_img_i = sitk.GetImageFromArray(eigen_array[..., i])
            copyImageInformation(strain, eigen_img_i)
            imgs.append(eigen_img_i)
        return imgs


def main():
    import argparse
    usage = "usage: given a dense deformation field, or areference image plus "
    usage += "a lis of ITK transformations, create the dense deformation"
    usage += " field.\n"
    usage += "NOTE: First Tx passed is the last Tx applied, for emxaple, the"
    usage += " orden should be [Affine, FFD]"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('-df',
                        '--dense_field',
                        help="Dense deformation field",
                        type=str,
                        default=None)
    parser.add_argument('-r',
                        '--ref',
                        help="Reference image",
                        type=str,
                        default=None)
    parser.add_argument('-sp',
                        '--save_prefix',
                        help="Save prefix file",
                        type=str)
    parser.add_argument('-ss',
                        '--save_strain',
                        help="Save strain as numpy",
                        action='store_true')
    parser.add_argument(
        '-t',
        '--transforms',
        nargs='+',
        help="List of transformations, for example [Affine, FFD]",
        type=str,
        default=None)

    parser.add_argument('-sf',
                        '--strain_form',
                        type=str,
                        default='infinitesimal',
                        choices=['lagragian', 'eulerian', 'infinitesimal'])

    args = parser.parse_args()
    if args.dense_field is None:
        # Read image
        ref_img = sitk.ReadImage(args.ref)
        # Read ITK Transfomrations
        tx = [sitk.ReadTransform(tx_name) for tx_name in args.transforms]
        tx2dfield = Transform2DenseField(tx, ref_img)
        dfield = tx2dfield.Execute()
    else:
        dfield = sitk.ReadImage(args.dense_field)

    # Dfiels is sitk and strain will be ITK
    strain_filter = StrainFilter(strain_form=StrainFilter.EULERIAN)
    strain = strain_filter.Execute(dfield)

    if args.save_strain:
        strain_array = itk.array_view_from_image(strain)
        np.save(args.save_prefix + '_strain.npy', strain_array)
    # np.load()
    # strain =

    order_by = itk.SymmetricEigenAnalysisEnums.EigenValueOrder_OrderByValue
    eigen_filter = ComputeEigenValues(order_by)
    eigen_images = eigen_filter.Execute(strain)
    for i, img_i in enumerate(eigen_images):
        fname_i = args.save_prefix + '_eigenval_%i.nrrd' % (i + 1)
        sitk.WriteImage(img_i, fname_i, True)


if __name__ == "__main__":
    main()
