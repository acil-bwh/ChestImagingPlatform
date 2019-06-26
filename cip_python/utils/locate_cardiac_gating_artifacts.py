import SimpleITK as sitk
import numpy as np
from cip_python.input_output import ImageReaderWriter


class LocateCardiacGatingArtifacts:

    def __init__(self, input_ct):
        self._input_ct = input_ct

    def locate_artifacts(self, thresh):

        input_image = sitk.GetArrayFromImage(self._input_ct)
        input_image = np.transpose(input_image, (2, 1, 0))
        gaussian_image = sitk.GetArrayFromImage(sitk.DiscreteGaussian(self._input_ct, 1.0))
        gaussian_image = np.transpose(gaussian_image, (2, 1, 0))

        diff_image = input_image - gaussian_image

        proj_std = np.zeros((diff_image.shape[0], diff_image.shape[2]))
        for i in range(diff_image.shape[0]):
            for k in range(diff_image.shape[2]):
                proj_std[i, k] = diff_image[i, :, k].std()

        proj_mean = np.zeros((proj_std.shape[1],))
        for i in range(proj_std.shape[1]):
            proj_mean[i] = proj_std[:, i].mean()

        diff_mean = np.diff(proj_mean)

        indexes = np.nonzero(diff_mean < -thresh)

        art_lm = np.zeros(input_image.shape, 'uint16')
        for idx in indexes:
            art_lm[:, :, idx] = 1

        return art_lm


if __name__ == "__main__":
    import optparse

    parser = optparse.OptionParser(description='Filter to convert CT signal into 2D projection and remove artifacts.')
    parser.add_option('--ict', help='Input CT image to analyze.', dest="i_ct", metavar='<string>')
    parser.add_option('--olm', help='Output CT image for artifacts removal.', dest="o_lm", metavar='<string>')
    parser.add_option('--th', help='Threshold value for sensitivity.', dest="th", metavar='<float>')
    parser.add_option('--invert', help='Invert the output mask.', dest="invert", action="store_true")

    options, args = parser.parse_args()

    image_read_write = ImageReaderWriter()
    input_ct = image_read_write.read(options.i_ct)
    threshold = float(options.th)

    la_class = LocateCardiacGatingArtifacts(input_ct)
    artifacts_lm = la_class.locate_artifacts(threshold)


    if options.invert == True:
        artifacts_lm = 1 - artifacts_lm

    output_lm = image_read_write.numpy_to_sitkImage(artifacts_lm, sitk_image_template=input_ct)
    image_read_write.write(output_lm, options.o_lm)







