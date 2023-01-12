"""
File: itk_sitk_support.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    This module provides a bridge between different ITK and SITK data types
"""

import numpy as np
import SimpleITK as sitk
import itk


def copyImageInformation(A, B):
    B.SetOrigin(np.array(A.GetOrigin()))
    B.SetSpacing(np.array(A.GetSpacing()))
    direction = A.GetDirection()
    if isinstance(A, sitk.Image) and (isinstance(B, itk.Image)
                                      or isinstance(B, itk.VectorImage)):
        direction = np.reshape(direction, [A.GetDimension()] * 2)
    elif isinstance(B, sitk.Image) and (isinstance(A, itk.Image)
                                        or isinstance(A, itk.VectorImage)):
        direction = np.array(direction).reshape(-1)

    B.SetDirection(direction)


def sitkImageToITK(simg):
    I = sitk.GetArrayFromImage(simg)
    is_vector = simg.GetNumberOfComponentsPerPixel() > 1
    img = itk.GetImageFromArray(I, is_vector=is_vector)
    copyImageInformation(simg, img)
    return img


def itkImageToSITK(img):
    I = itk.GetArrayFromImage(img)
    is_vector = img.GetNumberOfComponentsPerPixel() > 1
    simg = sitk.GetImageFromArray(I, isVector=is_vector)
    copyImageInformation(img, simg)
    return simg
