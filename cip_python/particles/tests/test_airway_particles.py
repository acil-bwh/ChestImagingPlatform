import os.path
import subprocess
import numpy as np
from numpy import sum, sqrt
import tempfile, shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pdb
from cip_python.particles.airway_particles import AirwayParticles
from cip_python.particles.particle_metrics import ParticleMetrics

def test_airway_particles():
  try:
    # Get the path to the this test so that we can reference the test data
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Set up the inputs to AirwayParticles
    input_ct = this_dir + '/../../../Testing/Data/Input/airwaygauss.nrrd'
    input_mask = this_dir + '/../../../Testing/Data/Input/airwaygauss_mask.nrrd'
      
    #tmp_dir = this_dir + '/../../../Testing/tmp/'
    tmp_dir = tempfile.mkdtemp()
    output_particles = os.path.join(tmp_dir,'airway_particles.vtk')
    print tmp_dir

    max_scale = 6.0
    live_th = 50.0
    seed_th = 40.0
    scale_samples = 5
    down_sample_rate = 1.0
    min_intensity = -1100
    max_intensity = -400

    # Generate the airway particles
    ap = AirwayParticles(input_ct, output_particles, tmp_dir, input_mask,
                         max_scale, live_th, seed_th, scale_samples,
                         down_sample_rate, min_intensity, max_intensity)
    ap.execute()
    
    # Read in the reference data set for comparison
    ref_reader = vtk.vtkPolyDataReader()
    ref_reader.SetFileName(this_dir +
                    '/../../../Testing/Data/Input/airway_particles.vtk')
    ref_reader.Update()

    # Now read in the output data set
    test_reader = vtk.vtkPolyDataReader()
    test_reader.SetFileName(output_particles)
    test_reader.Update()
    
    pm = ParticleMetrics(ref_reader.GetOutput(),
                         test_reader.GetOutput(), 'airway')
      
    assert pm.get_particles_dice() > 0.97, \
        "Airway particle Dice score lower than expected"

  finally:
    #Clear particles cache
    ap._clean_tmp_dir=True
    ap.clean_tmp_dir()
    shutil.rmtree(tmp_dir)