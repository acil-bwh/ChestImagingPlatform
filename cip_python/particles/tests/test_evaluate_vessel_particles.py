import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import pdb
from cip_python.particles.evaluate_vessel_particles import evaluate_vessel_particles

def test_evaluate_vessel_particles():
    # Set up the skeleton image
    skel = np.zeros([10, 10, 10])
    skel[5, 5, 5] = 1

    # Set up the polydata
    pts = vtk.vtkPoints()
    pts.InsertNextPoint(5., 5., 5.)
    poly = vtk.vtkPolyData()
    poly.SetPoints(pts)
    
    spacing = [1., 1., 1.]
    origin = [0., 0., 0.]
    irad = 1.5    
    score = evaluate_vessel_particles(skel, poly, spacing, origin, irad)
    assert score == 1.0, 'Expected score 1.0, but got score %s' % score

    # Update for an additional test
    skel[5, 5, 9] = 1    
    pts.InsertNextPoint(5., 7.4, 9.0)
    poly.SetPoints(pts)

    score = evaluate_vessel_particles(skel, poly, spacing, origin, irad)
    assert score == 1.0, 'Expected score 1.0, but got score %s' % score

    # Add a FN
    skel[5, 5, 0] = 1
    score = evaluate_vessel_particles(skel, poly, spacing, origin, irad)
    assert score == 0.8, 'Expected score 0.8, but got score %s' % score

    # Add a FP
    pts.InsertNextPoint(0, 0, 0)
    poly.SetPoints(pts)
    score = evaluate_vessel_particles(skel, poly, spacing, origin, irad)
    assert np.isclose(score, 2./3), 'Expected score 2/3, but got score %s'\
       % score
