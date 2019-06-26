"""
Read a not-isotropic DICOM series and generate an isotropic nrrd volume (discarding the information
in the slices that are closest to each other)
"""
import SimpleITK as sitk
import pydicom
import numpy as np
from argparse import ArgumentParser

class IsotropicDicomConverter(object):
    def run(self, dicom_input_folder, output_file_path):
        """
        Main method
        :param dicom_input_folder: str. Path to the DICOM series folder
        :param output_file_path: str. Path to the output nrrd volume
        """
        # Read the DICOM folder into a SimpleITK image
        reader = sitk.ImageSeriesReader()
        dicom_file_names = reader.GetGDCMSeriesFileNames(dicom_input_folder)
        reader.SetFileNames(dicom_file_names)
        sitk_image = reader.Execute()
        origin = sitk_image.GetOrigin()

        num_slices = len(dicom_file_names)
        assert num_slices >= 3, "The volume is too small. Corrupt DICOM folder?"

        # Get the distances that should be worked with
        pos0 = pydicom.dcmread(dicom_file_names[0]).SliceLocation
        pos1 = pydicom.dcmread(dicom_file_names[1]).SliceLocation
        pos2 = pydicom.dcmread(dicom_file_names[2]).SliceLocation
        # Get the expected distance between slices
        max_d = max(abs(pos1-pos0), abs(pos2-pos1))
        min_d = min(abs(pos1 - pos0), abs(pos2 - pos1))

        assert max_d != min_d, "The DICOM series seems to be isotropic"

        expected_d = max_d + min_d

        output_volume = np.zeros((num_slices // 2, sitk_image.GetSize()[0], sitk_image.GetSize()[1]), dtype=np.int16)
        # Read first slice
        ds = pydicom.dcmread(dicom_file_names[0])
        slope = float(ds.RescaleSlope)
        intercept = float(ds.RescaleIntercept)

        output_volume[0] = (ds.pixel_array * slope) + intercept

        prev_pos = pos0

        # Read each pair of slices
        for i in range(2, num_slices, 2):
            # Sanity check
            p = pydicom.dcmread(dicom_file_names[i]).SliceLocation
            assert abs(p - prev_pos) == expected_d,  \
                "Error in DICOM series. Position: {}. Expected distance: {}. Positions found: {}-{}".format(
                    i, expected_d, prev_pos, p
                )

            prev_pos = p
            output_volume[i // 2] = (pydicom.dcmread(dicom_file_names[i]).pixel_array * slope) + intercept



        # Convert to nrrd
        output_image = sitk.GetImageFromArray(output_volume)
        output_image.SetOrigin(origin)
        output_image.SetSpacing((sitk_image.GetSpacing()[0], sitk_image.GetSpacing()[1], expected_d))
        output_image.SetDirection(sitk_image.GetDirection())

        sitk.WriteImage(output_image, output_file_path)


if __name__ == '__main__':
    desc = "Read a non-isotropic DICOM series and generate an isotropic nrrd volume " \
           "(discarding the information in the slices that are closest to each other)"

    parser = ArgumentParser(description=desc)
    parser.add_argument('-i', '--input_dicom_folder', required=True, help='Input dicom folder')
    parser.add_argument('-o', '--output_file', required=True, help='Output full path to nrrd file')

    args = parser.parse_args()
    converter = IsotropicDicomConverter()
    converter.run(args.input_dicom_folder, args.output_file)