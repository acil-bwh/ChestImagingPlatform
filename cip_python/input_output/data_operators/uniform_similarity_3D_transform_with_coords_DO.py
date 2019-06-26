import SimpleITK as sitk
import numpy as np
from ...dcnn.data.data_processing import DataProcessing
from . import DataOperatorInterface

class UniformSimilarity3DTransformWithCoordsDataOperator(DataOperatorInterface):
    def __init__(self, output_size, min_translation, max_translation, min_scale, max_scale,
                 interpolator=sitk.sitkBSpline, default_pixel_value=0):
        """

        :param min_translation: float or 3-tuple (one elem per axis). Translation rate (relative to the image size)
        :param max_translation:
        :param min_scale:
        :param max_scale:
        """
        self.output_size = output_size
        self.min_translation = min_translation
        self.max_translation = max_translation
        self.min_scale = min_scale
        self.max_scale = max_scale

        # TODO: implement rotation
        self.interpolator = interpolator
        self.default_pixel_value = default_pixel_value

        self.translation = self.scale =  None

    def set_operation_parameters(self, translation, scale):
        self.translation = translation
        self.scale = scale
        # self.cardia_crop = cardia_crop

    def run(self, data, generate_parameters=True):
        # Sanity check
        if not isinstance(data, tuple):
            img = data
            coords = None

        else:
            img, coords = data

        assert isinstance(img, sitk.Image)
        assert coords is None or (isinstance(coords, np.ndarray)), "Coords must be an array or None"

        # if not isinstance(data, tuple) \
        #     or len(data) != 2 \
        #     or not isinstance(data[0], sitk.Image) \
        #     or not isinstance(data[1], np.ndarray):
        #         raise Exception("Expected a tuple of two elements: the SimpleITK image Andy the array of coordinates with start and size")
        if coords is not None:
            assert coords.shape[1] == 3, "Expected coords shape: (-1,3). Got {}".format(coords.shape)
        if generate_parameters:
            # Generate a 3D translation
            self.translation = self.scale = None
            if self.min_translation is not None:
                self.translation = np.random.uniform(self.min_translation, self.max_translation, 3)
                self.translation *= np.array(img.GetSize())
            if self.min_scale is not None:
                self.scale = np.random.uniform(self.min_scale, self.max_scale)

        transformed_image, transformed_coords = DataProcessing.similarity_3D_transform_with_coords(
            img, coords, self.output_size,
            self.translation, self.scale, interpolator=self.interpolator, default_pixel_value=self.default_pixel_value)
        return transformed_image, transformed_coords

