import vtk
import numpy as np
from cip_python.particles.transfer_particles_region_type_values import *

def test_transfer_particles_region_type_values():
    # Create labeled poly data
    labeled_points  = vtk.vtkPoints()
    labeled_points.InsertPoint(0, (0., 0., 0.))
    labeled_points.InsertPoint(1, (0., 0., 1.))

    labeled_region_type_arr = vtk.vtkFloatArray()
    labeled_region_type_arr.SetName('ChestRegionChestType')
    labeled_region_type_arr.InsertNextValue(771)
    labeled_region_type_arr.InsertNextValue(770)    

    labeled_hevec0_arr = vtk.vtkFloatArray()
    labeled_hevec0_arr.SetName('hevec0')
    labeled_hevec0_arr.SetNumberOfComponents(3)
    labeled_hevec0_arr.InsertNextTuple((0., 0., 1.))
    labeled_hevec0_arr.InsertNextTuple((0., 0., 1.))    

    labeled_hevec1_arr = vtk.vtkFloatArray()
    labeled_hevec1_arr.SetName('hevec1')
    labeled_hevec1_arr.SetNumberOfComponents(3)
    labeled_hevec1_arr.InsertNextTuple((0., 1., 0.))
    labeled_hevec1_arr.InsertNextTuple((0., 1., 0.))    

    labeled_hevec2_arr = vtk.vtkFloatArray()
    labeled_hevec2_arr.SetName('hevec2')
    labeled_hevec2_arr.SetNumberOfComponents(3)
    labeled_hevec2_arr.InsertNextTuple((1., 0., 0.))
    labeled_hevec2_arr.InsertNextTuple((1., 0., 0.))    

    labeled_scale_arr = vtk.vtkFloatArray()
    labeled_scale_arr.SetName('scale')
    labeled_scale_arr.InsertNextValue(1)
    labeled_scale_arr.InsertNextValue(1)    
    
    labeled_poly = vtk.vtkPolyData()
    labeled_poly.SetPoints(labeled_points)
    labeled_poly.GetPointData().AddArray(labeled_region_type_arr)
    labeled_poly.GetPointData().AddArray(labeled_hevec0_arr)
    labeled_poly.GetPointData().AddArray(labeled_hevec1_arr)
    labeled_poly.GetPointData().AddArray(labeled_hevec2_arr)
    labeled_poly.GetPointData().AddArray(labeled_scale_arr)        
    
    # Create unlabeled poly data
    unlabeled_points  = vtk.vtkPoints()
    unlabeled_points.InsertPoint(0, (0., 0., 0.2))
    unlabeled_points.InsertPoint(1, (0., 0., 1.2))
    unlabeled_points.InsertPoint(2, (0., 0., 3.))    

    unlabeled_region_type_arr = vtk.vtkFloatArray()
    unlabeled_region_type_arr.SetName('ChestRegionChestType')
    unlabeled_region_type_arr.InsertNextValue(0)
    unlabeled_region_type_arr.InsertNextValue(0)
    unlabeled_region_type_arr.InsertNextValue(0)        

    unlabeled_hevec0_arr = vtk.vtkFloatArray()
    unlabeled_hevec0_arr.SetName('hevec0')
    unlabeled_hevec0_arr.SetNumberOfComponents(3)
    unlabeled_hevec0_arr.InsertNextTuple((0., 0., 1.))
    unlabeled_hevec0_arr.InsertNextTuple((0., np.sin(np.pi*10/180),
                                          np.cos(np.pi*10/180)))    
    unlabeled_hevec0_arr.InsertNextTuple((0., 0., 1.))

    unlabeled_hevec1_arr = vtk.vtkFloatArray()
    unlabeled_hevec1_arr.SetName('hevec1')
    unlabeled_hevec1_arr.SetNumberOfComponents(3)
    unlabeled_hevec1_arr.InsertNextTuple((0., 1., 0.))
    unlabeled_hevec1_arr.InsertNextTuple((0., 1., 0.))
    unlabeled_hevec1_arr.InsertNextTuple((0., 1., 0.))        

    unlabeled_hevec2_arr = vtk.vtkFloatArray()
    unlabeled_hevec2_arr.SetName('hevec2')
    unlabeled_hevec2_arr.SetNumberOfComponents(3)
    unlabeled_hevec2_arr.InsertNextTuple((1., 0., 0.))
    unlabeled_hevec2_arr.InsertNextTuple((1., 0., 0.))
    unlabeled_hevec2_arr.InsertNextTuple((1., 0., 0.))    

    unlabeled_scale_arr = vtk.vtkFloatArray()
    unlabeled_scale_arr.SetName('scale')
    unlabeled_scale_arr.InsertNextValue(1)
    unlabeled_scale_arr.InsertNextValue(1)
    unlabeled_scale_arr.InsertNextValue(1)        
    
    unlabeled_poly = vtk.vtkPolyData()
    unlabeled_poly.SetPoints(unlabeled_points)
    unlabeled_poly.GetPointData().AddArray(unlabeled_region_type_arr)
    unlabeled_poly.GetPointData().AddArray(unlabeled_hevec0_arr)
    unlabeled_poly.GetPointData().AddArray(unlabeled_hevec1_arr)
    unlabeled_poly.GetPointData().AddArray(unlabeled_hevec2_arr)
    unlabeled_poly.GetPointData().AddArray(unlabeled_scale_arr)            

    out_poly = transfer_particles_region_type_values(unlabeled_poly,
        labeled_poly, v_angle_thresh=15, dist_thresh=1, num_candidates=2)

    assert out_poly.GetPointData().GetArray('ChestRegionChestType').\
      GetTuple(0)[0] == 771 and out_poly.GetPointData().\
      GetArray('ChestRegionChestType').GetTuple(1)[0] == 770 and \
        out_poly.GetPointData().GetArray('ChestRegionChestType').\
        GetTuple(2)[0] == 0, "Mislabeled region and type values"
