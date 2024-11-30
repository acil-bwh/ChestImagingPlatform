import numpy as np
import SimpleITK as sitk
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import scipy.ndimage.interpolation as scipy_interpolation

class DataProcessing(object):
    @classmethod
    def resample_image_itk(cls, image, output_size, output_type=None, interpolator=sitk.sitkBSpline):
        """
        Image resampling using ITK
        :param image: simpleITK image
        :param output_size: numpy array or tuple. Output size
        :param output_type: simpleITK output data type. If None, use the same as 'image'
        :param interpolator: simpleITK interpolator (default: BSpline)
        :return: tuple with simpleITK image and array with the resulting output spacing
        """
        if not isinstance(output_size, np.ndarray):
            output_size = np.array(output_size)
        factor = np.asarray(image.GetSize()) / output_size.astype(np.float32)
        output_spacing = np.asarray(image.GetSpacing()) * factor

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetSize(output_size.tolist())
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputSpacing(output_spacing)
        resampler.SetOutputPixelType(output_type if output_type is not None else image.GetPixelIDValue())
        resampler.SetOutputOrigin(image.GetOrigin())
        return resampler.Execute(image), output_spacing

    @classmethod
    def similarity_3D_transform_with_coords(cls, img, coords, output_size, translation, scale,
                                            interpolator=sitk.sitkBSpline, default_pixel_value=0.):
        """
        Apply a 3D similarity transform to an image and use the same transformation for a list of coordinates
        (rotation not implemented at the moment)
        :param img: simpleITK image
        :param coords: numpy array of coordinates (Nx3) or None
        :param output_size:
        :param scale:
        :param translation:
        :return: tuple with sitkImage, transformed_coords 
        """
        reference_image = sitk.Image(output_size, img.GetPixelIDValue())
        output_size_arr = np.array(output_size)
        reference_image.SetOrigin(img.GetOrigin())
        reference_image.SetDirection(img.GetDirection())
        spacing = (np.array(img.GetSize()) * np.array(img.GetSpacing())) / output_size_arr
        reference_image.SetSpacing(spacing)

        # Create the transformation
        tr = sitk.Similarity3DTransform()
        if translation is not None:
            tr.SetTranslation(translation)
        if scale is not None:
            tr.SetScale(scale)

        # Apply the transformation to the image
        img2 = sitk.Resample(img, reference_image, tr, interpolator, default_pixel_value)

        if coords is not None:
            # Apply the transformation to the coordinates
            transformed_coords = np.zeros_like(coords)
            for i in range(coords.shape[0]):            
                coords_ph = img.TransformContinuousIndexToPhysicalPoint(coords[i])
                coords_ph = tr.GetInverse().TransformPoint(coords_ph)
                transformed_coords[i] = np.array(img2.TransformPhysicalPointToContinuousIndex(coords_ph))
        else:
            transformed_coords = None
        
        return img2, transformed_coords

    @classmethod
    def scale_images(cls, img, output_size, return_scale_factors=False):
        """
        Scale an array that represents one or more images into a shape
        :param img: numpy array. It may contain one or multiple images
        :param output_size: tuple of int. Shape expected (including possibly the number of images)
        :param return_scale_factors: bool. If true, the result will be a tuple whose second values are the factors that
                                     were needed to downsample the images
        :return: numpy array rescaled or tuple with (array, factors)
        """
        img_size = np.array(img.shape)
        scale_factors = None
        if not np.array_equal(output_size, img_size):
            # The shape is the volume is different than the one expected by the network. We need to resize
            scale_factors = output_size / img_size
            # Reduce the volume to fit in the desired size
            img = scipy_interpolation.zoom(img, scale_factors)
        if return_scale_factors:
            return img, scale_factors
        return img

    @classmethod
    def standardization(cls, image_array, mean_value=-600, std_value=1.0, out=None):
        """
        Standarize an image substracting mean and dividing by variance
        :param image_array: image array
        :param mean_value: float. Image mean value. If None, ignore
        :param std_value: float. Image standard deviation value. If None, ignore
        :return: New numpy array unless 'out' parameter is used. If so, reference to that array
        """
        if out is None:
            # Build a new array (copy)
            image = image_array.astype(np.float32)
        else:
            # We will return a reference to out parameter
            image = out
            if id(out) != id(image_array):
                # The input and output arrays are different.
                # First, copy the source values, as we will apply the operations to image object
                image[:] = image_array[:]

        assert image.dtype == np.float32, "The out array must contain float32 elements, because the transformation will be performed in place"

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


    @classmethod
    def normalize_CT_image_intensity(cls, image_array, min_value=-300, max_value=700, min_output=0.0, max_output=1.0,
                                     out=None):
        """
        Threshold and adjust contrast range in a CT image.
        :param image_array: int numpy array (CT or partial CT image)
        :param min_value: int. Min threshold (everything below that value will be thresholded). If None, ignore
        :param max_value: int. Max threshold (everything below that value will be thresholded). If None, ignore
        :param min_output: float. Min out value
        :param max_output: float. Max out value
        :param out: numpy array. Array that will be used as an output
        :return: New numpy array unless 'out' parameter is used. If so, reference to that array
        """
        clip = min_value is not None or max_value is not None
        if min_value is None:
            min_value = np.min(image_array)
        if max_value is None:
            max_value = np.max(image_array)

        if out is None:
            # Build a new array (copy)
            image = image_array.astype(np.float32)
        else:
            # We will return a reference to out parameter
            image = out
            if id(out) != id(image_array):
                # The input and output arrays are different.
                # First, copy the source values, as we will apply the operations to image object
                image[:] = image_array[:]

        assert image.dtype == np.float32, "The out array must contain float32 elements, because the transformation will be performed in place"

        if clip:
            np.clip(image, min_value, max_value, image)

        # Change of range
        image -= min_value
        image /= (max_value - min_value)
        image *= (max_output - min_output)
        image += min_output

        return image


    @classmethod
    def elastic_transform(cls, image, alpha, sigma, fill_mode='constant', cval=0.):
        """
        Elastic deformation of images as described in  http://doi.ieeecomputersociety.org/10.1109/ICDAR.2003.1227801
        :param image: numpy array
        :param alpha: float
        :param sigma: float
        :param fill_mode: fill mode for gaussian filer. Default: constant value (cval)
        :param cval: float
        :return: numpy array. Image transformed
        """
        random_state = np.random.RandomState(None)
        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=fill_mode, cval=cval) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode=fill_mode, cval=cval) * alpha
        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        distorted_image = map_coordinates(image, indices, order=1).reshape(shape)
        return distorted_image