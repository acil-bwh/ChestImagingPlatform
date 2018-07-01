import numpy as np
import SimpleITK as sitk

def resample_image(image, output_size, output_type=sitk.sitkFloat32, interpolator=sitk.sitkBSpline):
    factor = np.asarray(image.GetSize()) / output_size.astype(np.float32)
    output_spacing = np.asarray(image.GetSpacing()) * factor

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(output_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputPixelType(output_type)
    return resampler.Execute(image)


def standardization(image_array):
    """
    Standarize an image substracting mean and dividing by variance
    :param image_array: image array
    :return: Standardized image array
    """
    image_array = image_array.astype(np.float32)
    MEAN = image_array.mean()
    STD = image_array.std()

    if STD <= 0.0001:
        STD = 1.0

    # Standardize image
    image_array -= MEAN
    image_array /= STD

    return image_array