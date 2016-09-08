import cip_python.nipype.interfaces.cip as cip
import os
import nipype.pipeline.engine as pe
from nipype.pipeline.engine import Workflow
from .. import CIPConventionManager as CM

#TODO: invalid class at the moment
class VesselParticlesWorkflow(Workflow):
    """This workflow produces vessel particles files for specified chest 
    regions. Output files are in .vtk format and have file names automatically
    constructed as 'cid_chestRegionVesselParticles.vtk', where 'cid' is taken
    to be the same prefix as the specified CT file name, and 'chestRegion' will
    be each of the specified chest regions (see param list). The output files
    will be saved in the same directory as the input CT file. 

    Parameters
    ----------
    ct_file_name : str
        The file name of the CT image (single file, 3D volume) in which to 
        identify seeds as possible vessel locations.

    tmp_dir : str
        Directory in which to store intermediate files used for this computation
        
    chest_regions : list of strings, optional
        A comma-separated list of chest regions within which to compute 
        particls. A separate particles file will be created for each region 
        specified. Specified chest regions must adhere to CIP conventions. By
        default, this list will contain 'LeftLung' and 'RightLung'.
    """
    def __init__(self, ct_file_name, tmp_dir, chest_regions=None):
        Workflow.__init__(self, 'VesselParticlesWorkflow')

        assert ct_file_name.rfind('.') != -1, "Unrecognized CT file name format"
        
        self._tmp_dir = tmp_dir
        self._cid = ct_file_name[max([ct_file_name.rfind('/'), 0])+1:\
                                 ct_file_name.rfind('.')]

        if ct_file_name.rfind('/') != -1:
            self._dir = ct_file_name[0:ct_file_name.rfind('/')]
        else:
            self._dir = '.'

        if vessel_seeds_mask_file_name is None:
            self._vessel_seeds_mask_file_name = \
              os.path.join(self._dir, self._cid + CM._vesselSeedsMask)
        else:
            self._vessel_seeds_mask_file_name = vessel_seeds_mask_file_name
            
        generate_partial_lung_label_map = \
          pe.Node(interface=cip.GeneratePartialLungLabelMap(), 
                  name='generate_partial_lung_label_map')
        generate_partial_lung_label_map.inputs.ct = ct_file_name
        # generate_partial_lung_label_map.inputs.
        
        extract_chest_label_map = \
          pe.Node(interface=cip.ExtractChestLabelMap(),
                  name='extract_chest_label_map')
        # extract_chest_label_map.inputs.outFileName =
        # extract_chest_label_map.inputs.
