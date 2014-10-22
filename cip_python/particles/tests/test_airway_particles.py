import os.path
import subprocess
import numpy as np
from numpy import sum, sqrt
import vtk
import pdb
from cip_python.particles.airway_particles import AirwayParticles

def test_airway_particles():
    # Get the path to the this test so that we can reference the test data
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Set up the inputs to AirwayParticles
    input_ct = this_dir + '/../../../Testing/Data/Input/airway.nrrd'
    output_particles = this_dir + '/../../../Testing/tmp/airway_particles.vtk'
    tmp_dir = this_dir + '/../../../Testing/tmp/'
    input_mask = None
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
    test_reader.SetFileName(this_dir +
                    '/../../../Testing/Data/Input/airway_particles.vtk')
    test_reader.Update()

    # The test passes provided that every particle in the reference data set
    # has a partner within an '_irad' distance of some particle in the test
    # data set and vice versa
    irad = ap._irad
    
    for i in xrange(0, ref_reader.GetOutput().GetNumberOfPoints()):
        ref_point = np.array(ref_reader.GetOutput().GetPoint(i))
        test_pass = False
        for j in xrange(0, test_reader.GetOutput().GetNumberOfPoints()):
            test_point = np.array(test_reader.GetOutput().GetPoint(j))
            dist = sqrt(sum((ref_point - test_point)**2))
            if dist <= irad:
                test_pass = True
                break
        assert test_pass, 'Airway particle has no match'

    for i in xrange(0, test_reader.GetOutput().GetNumberOfPoints()):
        test_point = np.array(test_reader.GetOutput().GetPoint(i))
        test_pass = False
        for j in xrange(0, ref_reader.GetOutput().GetNumberOfPoints()):
            ref_point = np.array(ref_reader.GetOutput().GetPoint(j))
            dist = sqrt(sum((ref_point - test_point)**2))
            if dist <= irad:
                test_pass = True
                break
        assert test_pass, 'Airway particle has no match'

    # Clean up the tmp directory. Note that this block should probably change
    # if this test is altered
    files = ['V-000-005.nrrd', 'V-001-005.nrrd', 'V-002-005.nrrd',
             'V-003-005.nrrd', 'V-004-005.nrrd', 'airway_particles.vtk',
             'ct-deconv.nrrd', 'hess.nrrd', 'heval0.nrrd', 'heval1.nrrd',
             'heval2.nrrd', 'hevec0.nrrd', 'hevec1.nrrd', 'hevec2.nrrd',
             'hmode.nrrd', 'pass1.nrrd', 'pass2.nrrd', 'pass3.nrrd',
             'val.nrrd', '2dd.nrrd', 'curvdir1.nrrd', 'curvdir2.nrrd',
             'flowlinecurv.nrrd', 'gausscurv.nrrd', 'gmag.nrrd',
             'gvec.nrrd', 'hf.nrrd', 'kappa1.nrrd', 'kappa2.nrrd',
             'lapl.nrrd', 'meancurv.nrrd', 'median.nrrd',
             'si.nrrd', 'st.nrrd', 'totalcurv.nrrd']

    for f in files:
        if os.path.isfile(tmp_dir + f):
            subprocess.call("rm " + tmp_dir + f, shell=True)
    
