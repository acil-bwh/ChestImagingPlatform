import os.path
import subprocess
import numpy as np
from numpy import sum, sqrt
import tempfile, shutil
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pdb
from cip_python.particles.vessel_particles import VesselParticles

def test_vessel_particles():
    try:
        # Get the path to the this test so that we can reference the test data
        this_dir = os.path.dirname(os.path.realpath(__file__))

        # Set up the inputs to AirwayParticles
        input_ct = this_dir + '/../../../Testing/Data/Input/vesselgauss.nrrd'
        input_mask = this_dir + '/../../../Testing/Data/Input/vessel_mask.nrrd'

        #tmp_dir = this_dir + '/../../../Testing/tmp/'
        tmp_dir = tempfile.mkdtemp()
        output_particles = os.path.join(tmp_dir,'vessel_particles.vtk')
        print tmp_dir
        max_scale = 6.0
        live_th = -100
        seed_th = -80
        scale_samples = 10
        down_sample_rate = 1.0
        min_intensity = -800
        max_intensity = 400

        # Generate the airway particles
        vp = VesselParticles(input_ct, output_particles, tmp_dir, 
                             input_mask, max_scale, live_th, seed_th, 
                             scale_samples, down_sample_rate,
                             min_intensity, max_intensity)    
        vp.execute()

        # Read in the reference data set for comparison
        ref_reader = vtk.vtkPolyDataReader()
        ref_reader.SetFileName(this_dir+'/../../../Testing/Data/Input/vessel_particles.vtk')
        ref_reader.Update()

        # Now read in the output data set
        test_reader = vtk.vtkPolyDataReader()
        test_reader.SetFileName(output_particles)
        test_reader.Update()
      
        
        # The test passes provided that every particle in the reference data set
        # has a partner within an '_irad' distance of some particle in the test
        # data set and vice versa
        irad = vp._irad

        ref_points = vtk_to_numpy(ref_reader.GetOutput().GetPoints().GetData())
        test_points = vtk_to_numpy(test_reader.GetOutput().GetPoints().GetData())

        #Compute pair-wise distance between all points (create 3D matrix to
        #perform the computation using vectorization)
        Np = ref_points.shape[0]
        Mp = test_points.shape[0]
        ref_tmp = np.expand_dims(ref_points,axis=2).repeat(Mp,axis=2)
        test_tmp = np.expand_dims(test_points,axis=2).transpose([2,1,0]).repeat(Np,axis=0)

        distance = np.sqrt(np.sum((ref_tmp-test_tmp)**2,axis=1)).squeeze()

        #Distance matrix should be perfectly symmetric
        # \|D-D.T\| should be zero for perfect matching
        #We can relax a bit this condition as particles have a small jitter effect.
        mindist_rows = np.sort(distance,axis=0)[0,:]
        mindist_cols = np.sort(distance,axis=1)[:,0]
        dist_test = (np.mean(mindist_rows[mindist_rows>0]) + np.mean(mindist_cols[mindist_cols>0]))*0.5
        test_pass = False
        if dist_test <= irad or dist_test == None:
          test_pass = True
      
        assert test_pass, 'Vessel particle has no match ' + str(dist_test) + ' ' + str(Np) + ' ' + str(Mp)

        #Compute differences in scale values
            

    finally:
        #Clear particles cache
        vp._clean_tmp_dir=True
        vp.clean_tmp_dir()
        shutil.rmtree(tmp_dir)
