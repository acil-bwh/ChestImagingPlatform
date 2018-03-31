import os.path
import pandas as pd
import vtk
from cip_python.phenotypes.vasculature_phenotypes import *
import cip_python.common as common

vessel_p_name = common.Paths.testing_file_path('vessel_particles.vtk')

vessel_reader = vtk.vtkPolyDataReader()
vessel_reader.SetFileName(vessel_p_name)
vessel_reader.Update()

vessel = vessel_reader.GetOutput()

num_points=vessel.GetNumberOfPoints()
chest_region=vtk.vtkShortArray()
chest_region.SetNumberOfTuples(num_points)

for kk in xrange(num_points):
  chest_region.SetValue(kk,1)

chest_region.SetName('ChestRegionChestType')
vessel.GetPointData().AddArray(chest_region)

def test_execute():

  v_pheno = VasculaturePhenotypes(chest_regions=['WholeLung'])
  out=v_pheno.execute(vessel,cid='None')

  df=out[0]

  #Interparticle distance is computed using a random set of points and the phenotype value
  #changes a bit between trials due to this random effect this is why we allow a 1% relative tolerance.
  assert np.isclose(float(df['TBV'].iloc[0]),float(4493.1982078),rtol=0.01), 'Phenotype not as expected'
  assert np.isclose(float(df['BV5'].iloc[0]),float(15.6746190252),rtol=0.01), 'Phenotypes not as expected'

