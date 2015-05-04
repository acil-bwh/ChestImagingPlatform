import os.path
import subprocess
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pdb
from cip_python.particles.particle_metrics import ParticleMetrics

def test_particle_metrics():
    # Get the path to the this test so that we can reference the test data
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Read in the reference data set for comparison
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(this_dir + \
        '/../../../Testing/Data/Input/vessel_particles.vtk')
    reader.Update()

    pm = ParticleMetrics(reader.GetOutput(), 
                         reader.GetOutput(), 'vessel')

    assert pm.get_particles_dice() == 1., \
      "Vessel particle Dice score lower than expected"
        
