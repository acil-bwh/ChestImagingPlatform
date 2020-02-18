"""
Read a not-isotropic DICOM series and generate an isotropic nrrd volume (discarding the information
in the slices that are closest to each other)
"""
import os
import shutil
import pydicom
import SimpleITK as sitk
import numpy as np
from argparse import ArgumentParser

from cip_python.input_output import ImageReaderWriter


class IodineMapDicomConverter(object):
    def run(self, dicom_input_folder, input_ct_file, output_file_path, tmp_folder, feet_first=False):
        """
        Main method
        :param dicom_input_folder: str. Path to the DICOM series folder
        :param output_file_path: str. Path to the output nrrd volume
        :param feet_first: bool. Flag to indicate that scan is feet first
        """
        # Re-name DICOM file using SOPInstanceUID tag

        if not os.path.isdir(tmp_folder):
            os.mkdir(tmp_folder)

        dcm_names = os.listdir(dicom_input_folder)
        for nn in dcm_names:
            in_name = os.path.join(dicom_input_folder, nn)
            ds = pydicom.dcmread(in_name)
            instance_tag = ds.SOPInstanceUID
            out_name = os.path.join(tmp_folder, '{}.dcm'.format(instance_tag))
            ds.save_as(out_name)

        # Read the DICOM folder into a SimpleITK image
        dicom_reader = sitk.ImageSeriesReader()
        dicom_file_names = dicom_reader.GetGDCMSeriesFileNames(tmp_folder)
        dicom_reader.SetFileNames(dicom_file_names)
        imap_sitk= dicom_reader.Execute()

        if feet_first:
            imap_sitk = sitk.Flip(imap_sitk, [False, False, True], True)

        io_sitk = ImageReaderWriter()
        ct_sitk = io_sitk.read(input_ct_file)

        if imap_sitk.GetSize()[0] > ct_sitk.GetSize()[0]:
            imap_sitk = imap_sitk[1:-1, 1:-1, :]
        elif imap_sitk.GetSize()[0] < ct_sitk.GetSize()[0]:
            imap_np = sitk.GetArrayFromImage(imap_sitk).transpose([2, 1, 0, 3])
            diff_x = (ct_sitk.GetSize()[0] - imap_sitk.GetSize()[0]) // 2
            diff_y = (ct_sitk.GetSize()[1] - imap_sitk.GetSize()[1]) // 2
            imap_np = np.pad(imap_np, ((diff_x, diff_x), (diff_y, diff_y), (0, 0), (0, 0)), 'constant')
            imap_sitk = sitk.GetImageFromArray(imap_np.transpose([2, 1, 0, 3]))

        imap_sitk.CopyInformation(ct_sitk)
        io_sitk.write(imap_sitk, output_file_path)

        shutil.rmtree(tmp_folder)


if __name__ == '__main__':
    desc = "Read a iodine map DICOM series and generate a nrrd volume " \
           "(fixing error in DICOM file names)"

    parser = ArgumentParser(description=desc)
    parser.add_argument('-i', required=True, help='Input dicom folder', dest='input_dicom_folder')
    parser.add_argument('-ict', required=True, help='Input CT (.nrrd) file', dest='input_ct_file')
    parser.add_argument('-o', required=True, help='Output full path to nrrd file', dest='output_file')
    parser.add_argument('-tmp', required=True, help='Temp folder to save re-named dicom files', dest='tmp_folder')
    parser.add_argument('--ff', help='Flag to indicate that scan is feet first', action="store_true",
                        dest="feet_first")

    args = parser.parse_args()
    converter = IodineMapDicomConverter()
    input_dicom_folder = os.path.realpath(args.input_dicom_folder)
    tmp_folder = os.path.realpath(args.tmp_folder)
    converter.run(input_dicom_folder, args.input_ct_file, args.output_file, tmp_folder, args.feet_first)
