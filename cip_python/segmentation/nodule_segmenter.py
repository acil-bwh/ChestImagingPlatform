import subprocess
import SimpleITK as sitk
import xml.etree.ElementTree as ET
from optparse import OptionParser
from cip_python.common import ChestConventions

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

    @staticmethod
    def get_nodule_information(xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        nodules_id = []
        nodules_type = []
        nodules_seed = []

        coord_syst = root.find('CoordinateSystem').text
        cc = ChestConventions()
        #NoduleChestTypes=[86,87,88]
        NoduleChestTypes=['Nodule','BenignNodule','MalignantNodule']

        for n in root.findall('Point'):
            chesttype=int(n.find('ChestType').text)
            chesttype_name=cc.GetChestTypeNameFromValue(chesttype)
            if chesttype_name in NoduleChestTypes:
              n_id = n.find('Id').text
              nodules_id.append(n_id)
              #t = n.find('Description').text
              #nodules_type.append(t)
              nodules_type.append(chesttype_name)
              coordinate = n.find('Coordinate')
              seed = []
              for s in coordinate.findall('value'):
                seed.append(s.text)
              nodules_seed.append(seed)

        return coord_syst, nodules_id, nodules_type, nodules_seed

if __name__ == "__main__":
    desc = """Lung nodule segmentation tool"""

    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
        help='Input CT file', dest='in_ct', metavar='<string>',
        default=None)
    parser.add_option('--xml',
        help='XML file containing nodule information for the input ct.', dest='xml_file',
        metavar='<string>', default=None)
    parser.add_option('--n_lm',
        help='Output nodule labelmap.', dest='n_lm',
        metavar='<string>', default=None)
    parser.add_option('--max_rad',
                        help='Maximum radius (mm) for the lesion. Recommended: 30 mm \
                        for humans and 3 mm for small animals',
                        dest='max_rad', metavar='<float>', default=30.0)
    parser.add_option('--th',
                        help='Threshold value for nodule segmentation. All the voxels above the threshold will be \
                              considered nodule',
                        dest='segm_th', metavar='<float>', default=0.0)
    parser.add_option('--n_id',
                        help='Nodule id to select from the xml for the segmentation',
                        dest='n_id', metavar='<int>', default=0)
    (options, args) = parser.parse_args()

    input_ct = sitk.ReadImage(options.in_ct)

    coord_system, ids, types, seeds = NoduleSegmenter.get_nodule_information(options.xml_file)

    if len(ids) > 1:
        #More than one nodule set name
        print("XML file contains more than one nodule")

    seed_point=seeds[options.n_id]
    nodule_segmenter = NoduleSegmenter(input_ct, options.in_ct, float(options.max_rad), seed_point,
                                       options.n_lm, float(options.segm_th))
    nodule_segmenter.segment_nodule()

