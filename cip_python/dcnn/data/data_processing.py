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
    def standardization(image_array, mean_value=-600, std_value=1.0, out=None):
        """
        Standarize an image substracting mean and dividing by variance
        :param image_array: image array
        :param mean_value: float. Image mean value. If None, ignore
        :param std_value: float. Image standard deviation value. If None, ignore
        :return: Standardized image array
        """
        if out is None:
            # Get a copy of a new image
            image = image_array.astype(np.float32)
        else:
            # We will return a reference to out parameter
            image = out
            if id(out) != id(image_array):
                # The input and output arrays are different.
                # First, copy the source values, as we will apply the operations to image object
                image[:] = image_array[:]

        assert image.dtype == np.float32, "The out array must contain float32 elements"

        if mean_value is None:
            mean_value = image.mean()
        if std_value is None:
            std_value = image.std()
            if std_value <= 0.0001:
                std_value = 1.0

        # Standardize image
        image -= mean_value
        image /= std_value

        return image

    @staticmethod
    def normalize_CT_image_intensity(image_array, min_value=-300, max_value=700, min_output=0.0, max_output=1.0,
                                     out=None):
        """
        Threshold and adjust contrast range in a CT image.
        :param image_array: int numpy array (CT or partial CT image)
        :param min_value: int. Min threshold (everything below that value will be thresholded). If None, ignore
        :param max_value: int. Max threshold (everything below that value will be thresholded). If None, ignore
        :param min_output: float. Min out value
        :param max_output: float. Max out value
        :return: None if in_place==True. Otherwise, float numpy array with adapted intensity
        """
        clip = min_value is not None or max_value is not None
        if min_value is None:
            min_value = np.min(image_array)
        if max_value is None:
            max_value = np.max(image_array)
        if clip:
            np.clip(image_array, min_value, max_value, image_array)

        if out is None:
            # Get a copy of a new image
            image = image_array.astype(np.float32)
        else:
            # We will return a reference to out parameter
            image = out
            if id(out) != id(image_array):
                # The input and output arrays are different.
                # First, copy the source values, as we will apply the operations to image object
                image[:] = image_array[:]

        assert image.dtype == np.float32, "The out array must contain float32 elements, because the transformation will be performed in place"

        # Change of range
        image -= min_value
        image /= (max_value - min_value)
        image *= (max_output - min_output)
        image += min_output

        return image


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