from optparse import OptionParser
import cip_python.nipype.interfaces.cip as cip
import cip_python.nipype.interfaces.unu as unu
import cip_python.nipype.interfaces.cip.cip_pythonWrap as cip_python_interfaces
from cip_python.nipype.cip_convention_manager import CIPConventionManager as CM
import nipype.pipeline.engine as pe         # the workflow and node wrappers
from nipype.pipeline.engine import Workflow
import tempfile, shutil
import pydot
import sys
import os 
import pdb

class GenerateMedianFilteredImageWorkflow(Workflow):
    """This workflow generates a median filtered image.

    Parameters
    ----------
    ct_file_name : str
        The file name of the CT image (single file, 3D volume) to be filtered

    median_filtered_file_name : str
        The output median filtered file name
    """
    def __init__(self, ct_file_name, median_filtered_file_name):
        Workflow.__init__(self, 'GenerateMedianFilteredImageWorkflow')

        if False:
            assert ct_file_name.rfind('.') != -1, "Unrecognized CT file name format"
            
            self._cid = ct_file_name[max([ct_file_name.rfind('/'), 0])+1:\
                                     ct_file_name.rfind('.')]
    
            if ct_file_name.rfind('/') != -1:
                self._dir = ct_file_name[0:ct_file_name.rfind('/')]
            else:
                self._dir = '.'

        generate_median_filtered_image1 = \
          pe.Node(interface=cip.GenerateMedianFilteredImage(),
                  name='generate_median_filtered_image1') 
        generate_median_filtered_image1.inputs.outputFile = \
          '/Users/jross/tmp/foo_nipype/test_cache/m1.nhdr'
        generate_median_filtered_image1.inputs.inputFile = \
          '/Users/jross/Downloads/ChestImagingPlatform/Testing/Data/Input/vessel.nrrd'          
        generate_median_filtered_image1.inputs.Radius = 1

        generate_median_filtered_image2 = \
          pe.Node(interface=cip.GenerateMedianFilteredImage(),
                  name='generate_median_filtered_image2') 
        generate_median_filtered_image2.inputs.outputFile = \
          '/Users/jross/tmp/foo_nipype/test_cache/m2.nhdr'
        generate_median_filtered_image2.inputs.Radius = 1
        
        # Set up the workflow connections        
        self.connect(generate_median_filtered_image1, 'outputFile', 
                     generate_median_filtered_image2, 'inputFile')
        
if __name__ == "__main__":
    desc = """This workflow generates a median filtered image"""

    parser = OptionParser(description=desc)
    parser.add_option('--in_ct', help='The file name of the CT image (single \
                      file, 3D volume) to be filtered', dest='in_ct', 
                      metavar='<string>',  default=None)
    parser.add_option('--out', help='File name of output vessel seeds mask. \
                      If none is specified, a file name will be created using \
                      the CT file name prefix with the suffix \
                      _vesselSeedsMask.nhdr. The seeds mask indicates possible \
                      vessel loctions with the Vessel chest type label.', 
                      dest='out', metavar='<string>', default=None)                      

    (op, args) = parser.parse_args()
    
    tmp_dir = tempfile.mkdtemp()
    wf = VesselParticlesMaskWorkflow(op.in_ct, op.in_lm, tmp_dir, op.out)
    wf.run()
    shutil.rmtree(tmp_dir)
    
