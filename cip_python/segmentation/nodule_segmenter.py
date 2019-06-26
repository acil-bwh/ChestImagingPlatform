import SimpleITK as sitk
import subprocess


class NoduleSegmenter:

    def __init__(self, input_ct, input_ct_filename, max_radius, seed, output_lm, threshold):

        self._input_ct = input_ct
        self._input_ct_filename = input_ct_filename
        self._max_radius = max_radius
        self._seed = seed
        self._output_lm = output_lm
        self._threshold = threshold

    def segment_nodule(self):
        """ Run the nodule segmentation through a CLI
        """
        tmpCommand = "GenerateLesionSegmentation -i %(in)s -o %(out)s --seeds %(sd)s --maximumRadius %(maxrad)f -f"
        tmpCommand = tmpCommand % {'in': self._input_ct_filename, 'out': self._output_lm, 'sd': ",".join(map(str,self._seed)),
                                   'maxrad': self._max_radius}
        # tmpCommand = os.path.join(path['CIP_PATH'], tmpCommand)
        subprocess.call(tmpCommand, shell=True)

        nodule_segm_image = sitk.ReadImage(self._output_lm)
        nodule_segm_image = sitk.GetArrayFromImage(nodule_segm_image)
        nodule_segm_image[nodule_segm_image > self._threshold] = 1
        nodule_segm_image[nodule_segm_image < self._threshold] = 0
        nodule_segm_image=nodule_segm_image.astype('uint16') #Casting segmentation to unsigned short
        sitkImage = sitk.GetImageFromArray(nodule_segm_image)
        sitkImage.SetSpacing(self._input_ct.GetSpacing())
        sitkImage.SetOrigin(self._input_ct.GetOrigin())
        sitk.WriteImage(sitkImage, self._output_lm,True)