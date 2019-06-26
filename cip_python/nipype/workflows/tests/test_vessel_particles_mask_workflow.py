import os.path
import tempfile, shutil
import nrrd
from cip_python.utils import compute_dice_coefficient
from cip_python.nipype.workflows import VesselParticlesMaskWorkflow

from cip_python.common import Paths

def test_vessel_particles_mask_workflow():
    # Set up the inputs and run the workflow
    ct_file_name = Paths.testing_file_path('vessel.nrrd')
    label_map_file_name = Paths.testing_file_path('vessel_volumeMask.nrrd')
    seeds_mask_file_name = Paths.testing_file_path('vessel_vesselSeedsMask.nrrd')
    tmp_dir = tempfile.mkdtemp()
    vessel_seeds_mask_file_name = os.path.join(tmp_dir, 'vesselSeedsMask.nrrd')

    wf = VesselParticlesMaskWorkflow(ct_file_name, label_map_file_name, 
                                     tmp_dir, vessel_seeds_mask_file_name)
    wf.run()

    ref, ref_header = nrrd.read(seeds_mask_file_name)
    test, test_header = nrrd.read(vessel_seeds_mask_file_name)    
    dice = compute_dice_coefficient(ref, test, 1)
    shutil.rmtree(tmp_dir)
    
    assert dice > 0.995, "Dice coefficient lower than expected"
