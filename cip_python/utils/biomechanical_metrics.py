import SimpleITK as sitk
import numpy as np
from optparse import OptionParser
from os.path import isfile



class BiomechanicalMetrics():
    def __init__(self):
        pass

    @staticmethod
    def calculate_deformationtensor_metrics(l1, l2, l3, mask):
        '''Inputs:
          l1: largest eigenvalue from deformation tensor (sitkImage)
          l2: middle eigenvalue from deformation tensor (sitkImage)
          l3: smallest eigenvalue from deformation tensor (sitkImage)
          mask: lung mask (sitkImage)

          Outputs:
          J: Jacobian metric image (sitkImage)
          ADI: ADI image (sitkImage)
          SRI: SRI metric image (sitkImage)
          '''

        #
        # convert to numpy
        l1_np = sitk.GetArrayFromImage(l1)
        l2_np = sitk.GetArrayFromImage(l2)
        l3_np = sitk.GetArrayFromImage(l3)

        # apply sqrt to the eigenvalues
        l1_np = np.sqrt(l1_np)
        l2_np = np.sqrt(l2_np)
        l3_np = np.sqrt(l3_np)

        # calculate Jacobian index, note: np.* calculates multiplicates arrays, not matrix (correct)
        J_np = l1_np*l2_np*l3_np

        #calculate Anisotropic Deformation Index (ADI)
        ADI_np = np.sqrt(((l1_np - l2_np)/l2_np)**2 + ((l2_np - l3_np)/l3_np)**2)

        #calculate Slab-Rod Index
        SRI_np = (np.arctan((l3_np*(l1_np-l2_np))/(l2_np*(l2_np-l3_np))))/(np.pi/2)

        # apply mask
        mask_np = sitk.GetArrayFromImage(mask)

        J_m = J_np*mask_np
        ADI_m = ADI_np*mask_np
        SRI_m = SRI_np*mask_np

        # Save indices as nrrd.
        J = sitk.GetImageFromArray(J_m)
        ADI = sitk.GetImageFromArray(ADI_m)
        SRI = sitk.GetImageFromArray(SRI_m)

        J.CopyInformation(l1)
        ADI.CopyInformation(l1)
        SRI.CopyInformation(l1)

        return J,ADI,SRI


def main():
    usage = "usage: given the 3 eigenvalues of the stain tensor, calculates the J, ADI and SRI indices"
    parser = OptionParser(usage)

    parser.add_option('--l1',\
                      help='first eigenvalue matrix',\
                      dest='l1',metavar='<string>', default=None)
    parser.add_option('--l2',\
                      help='second eigenvalue matrix',\
                      dest='l2', default=None)
    parser.add_option('--l3',\
                      help='third eigenvalue matrix',\
                      dest='l3', default=None)
    parser.add_option('--mask',\
                      help='lung label map mask',\
                      dest='mask')
    parser.add_option('--J',\
                      help='name the J output file',\
                      dest='J_path', metavar='<string>', default='J.nrrd')
    parser.add_option('--ADI',\
                      help='name the ADI output file',\
                      dest='ADI_path', metavar='<string>', default='ADI.nrrd')
    parser.add_option('--SRI',\
                      help='name the SRI output file',\
                      dest='SRI_path', metavar='<string>', default='SRI.nrrd')

    (options, args) = parser.parse_args()



    assert isfile(options.l1), 'L1 file does not exist'
    assert isfile(options.l2), 'L2 file does not exist'
    assert isfile(options.l3), 'L3 file does not exist'
    assert isfile(options.mask), 'Mask file does not exist'

    print options.l1
    if (options.l1 is not None) and (options.l2 is not None) and (options.l3 is not None):

        #Read data
        l1 = sitk.ReadImage(options.l1)
        l2 = sitk.ReadImage(options.l2)
        l3 = sitk.ReadImage(options.l3)
        mask = sitk.ReadImage(options.mask)

        (J, ADI, SRI)=BiomechanicalMetrics.calculate_deformationtensor_metrics(l1,l2,l3,mask)

        sitk.WriteImage(J, options.J_path, True)
        sitk.WriteImage(ADI, options.ADI_path, True)
        sitk.WriteImage(SRI, options.SRI_path, True)

if __name__ == "__main__":
    main()