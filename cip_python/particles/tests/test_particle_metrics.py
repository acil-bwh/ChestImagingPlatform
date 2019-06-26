import vtk

from cip_python.particles.particle_metrics import ParticleMetrics
from cip_python.common import Paths

def test_particle_metrics():
    # Read in the reference data set for comparison
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(Paths.testing_file_path('vessel_particles.vtk'))
    reader.Update()

    pm = ParticleMetrics(reader.GetOutput(), 
                         reader.GetOutput(), 'vessel')

    assert pm.get_particles_dice() == 1., \
      "Vessel particle Dice score lower than expected"
        
