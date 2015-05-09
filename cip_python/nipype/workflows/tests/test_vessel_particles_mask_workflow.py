import os.path
import subprocess
import numpy as np
import tempfile, shutil
import pdb
import nrrd
from cip_python.utils.compute_dice_coefficient import compute_dice_coefficient
from cip_python.nipype.workflows.vessel_particles_mask_workflow \
  import VesselParticlesMaskWorkflow

def test_vessel_particles_mask_workflow():
    # Get the path to the this test so that we can reference the test data
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Set up the inputs and run the workflow
    ct_file_name = this_dir + '/../../../../Testing/Data/Input/vessel.nrrd'
    label_map_file_name = \
      this_dir + '/../../../../Testing/Data/Input/vessel_volumeMask.nrrd'
    seeds_mask_file_name = \
      this_dir + '/../../../../Testing/Data/Input/vessel_vesselSeedsMask.nrrd'
    tmp_dir = tempfile.mkdtemp()
    vessel_seeds_mask_file_name = os.path.join(tmp_dir, 'vesselSeedsMask.nrrd')

    wf = VesselParticlesMaskWorkflow(ct_file_name, label_map_file_name, 
                                     tmp_dir, vessel_seeds_mask_file_name)
    wf.run()

    ref, ref_header = nrrd.read(seeds_mask_file_name)
    test, test_header = nrrd.read(vessel_seeds_mask_file_name)    
    dice = compute_dice_coefficient(ref, test, 1)
    shutil.rmtree(tmp_dir)
    
    assert dice > 0.69, "Dice coefficient lower than expected"
