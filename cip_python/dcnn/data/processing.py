import numpy as np
import SimpleITK as sitk

class DataProcessing(object):
    @staticmethod
    def resample_image_itk(image, output_size, output_type=sitk.sitkFloat32, interpolator=sitk.sitkBSpline):
        """
        Image resampling using ITK
        :param image: simpleITK image
        :param output_size: int-tuple. Output size
        :param output_type: output data type
        :param interpolator: simpleITK interpolator (default: BSpline)
        :return: simpleITK image
        """
        factor = np.asarray(image.GetSize()) / output_size.astype(np.float32)
        output_spacing = np.asarray(image.GetSpacing()) * factor

        print (image.GetSpacing(), output_spacing)

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(output_size)
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputPixelType(output_type)
        return resampler.Execute(image), output_spacing

    @staticmethod
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

    @staticmethod
    def elastic_transform(image, alpha, sigma, fill_mode='constant', cval=0.):
        """
        Elastic deformation of images as described in  http://doi.ieeecomputersociety.org/10.1109/ICDAR.2003.1227801
        :param image: numpy array
        :param alpha: float
        :param sigma: float
        :param fill_mode: fill mode for gaussian filer. Default: constant value (cval)
        :param cval: float
        :return: numpy array. Image transformed
        """
        from scipy.ndimage.filters import gaussian_filter
        from scipy.ndimage.interpolation import map_coordinates
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=fill_mode, cval=cval) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=fill_mode, cval=cval) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        distorted_image = map_coordinates(image, indices, order=1).reshape(shape)
        return distorted_image