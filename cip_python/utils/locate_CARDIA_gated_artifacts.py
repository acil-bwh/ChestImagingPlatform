import SimpleITK as sitk
import numpy as np


class LocateCARDIAGatedArtifacts:

    def __init__(self, input_ct):
        self._input_ct = input_ct

    def locate_artifacts(self):

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

        indexes = np.nonzero(diff_mean < -0.5)

        art_lm = np.ones(input_image.shape, 'int32')
        for idx in indexes:
            art_lm[:, :, idx] = 0

        return art_lm


if __name__ == "__main__":
    import optparse

    parser = optparse.OptionParser(description='Filter to convert CT signal into 2D projection and remove artifacts.')
    parser.add_option('--ict', help='Input CT image to analyze.', dest="i_ct", metavar='<string>')
    parser.add_option('--olm', help='Output CT image for artifacts removal.', dest="o_lm", metavar='<string>')
    options, args = parser.parse_args()

    input_ct = sitk.ReadImage(options.i_ct)

    la_class = LocateCARDIAGatedArtifacts(input_ct)
    artifacts_lm = la_class.locate_artifacts()
    artifacts_lm = np.transpose(artifacts_lm, (2, 1, 0))

    metainfo = dict()
    metainfo['space origin'] = input_ct.GetOrigin()
    metainfo['spacing'] = input_ct.GetSpacing()
    metainfo['space directions'] = input_ct.GetDirection()

    output_lm = sitk.GetImageFromArray(artifacts_lm)

    output_lm.SetSpacing(metainfo['spacing'])
    output_lm.SetOrigin(metainfo['space origin'])
    output_lm.SetDirection(metainfo['space directions'])

    sitk.WriteImage(output_lm, options.o_lm)








