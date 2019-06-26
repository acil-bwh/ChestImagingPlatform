import numpy as np
from cip_python.common import ChestConventions
import vtk
import pandas as pd
import pdb
from cip_python.phenotypes.fissure_phenotypes import FissurePhenotypes

def test_main():
    # First create a test image
    conventions = ChestConventions()
    LUL = conventions.GetChestRegionValueFromName('LeftSuperiorLobe')
    LLL = conventions.GetChestRegionValueFromName('LeftInferiorLobe')
    F = conventions.GetChestTypeValueFromName('ObliqueFissure')
    LLL_F = conventions.GetValueFromChestRegionAndType(LLL, F)
    test_im = LLL*np.ones([1, 8, 8])
    test_im[0, 0, 2::] = LUL
    test_im[0, 1, 2::] = LUL
    test_im[0, 2, 2::] = LUL
    test_im[0, 3, 3::] = LUL
    test_im[0, 4, 4::] = LUL
    test_im[0, 5, 5::] = LUL
    test_im[0, 6, 5::] = LUL
    test_im[0, 7, 5::] = LUL

    test_im[0, 0, 1] = LLL_F
    test_im[0, 1, 1] = LLL_F
    test_im[0, 6, 4] = LLL_F
    test_im[0, 7, 4] = LLL_F
    
    completeness_phenos = FissurePhenotypes()
    df = completeness_phenos.execute(test_im, np.array([0, 0, 0]),
        np.array([1, 1, 2]), 'foo')

    assert df.Completeness.values[0] < 0.5, \
      "Completeness measure should be less than 0.5"

    df = completeness_phenos.execute(test_im, np.array([0, 0, 0]),
        np.array([1, 1, 2]), 'foo', completeness_type='domain')

    assert df.Completeness.values[0] == 0.5, \
      "Completeness measure should equal 0.5"
    
    test_im = LLL*np.ones([1, 8, 8])
    test_im[0, 0, 2::] = LUL
    test_im[0, 1, 2::] = LUL
    test_im[0, 2, 2::] = LUL
    test_im[0, 3, 3::] = LUL
    test_im[0, 4, 4::] = LUL
    test_im[0, 5, 5::] = LUL
    test_im[0, 6, 5::] = LUL
    test_im[0, 7, 5::] = LUL

    test_im[0, 2, 1] = LLL_F
    test_im[0, 3, 2] = LLL_F
    test_im[0, 4, 3] = LLL_F
    test_im[0, 5, 4] = LLL_F

    df = completeness_phenos.execute(test_im, np.array([0, 0, 0]),
        np.array([1, 1, 2]), 'foo')

    assert df.Completeness.values[0] > 0.5, \
        "Completeness measure should be greater than 0.5"    

    df = completeness_phenos.execute(test_im, np.array([0, 0, 0]),
        np.array([1, 1, 2]), 'foo', completeness_type='domain')

    assert df.Completeness.values[0] == 0.5, \
      "Completeness measure should equal 0.5"        

    test_im = LLL*np.ones([2, 8, 8])
    test_im[0, 0, 2::] = LUL; test_im[1, 0, 2::] = LUL
    test_im[0, 1, 2::] = LUL; test_im[1, 1, 2::] = LUL
    test_im[0, 2, 2::] = LUL; test_im[1, 2, 2::] = LUL
    test_im[0, 3, 3::] = LUL; test_im[1, 3, 3::] = LUL
    test_im[0, 4, 4::] = LUL; test_im[1, 4, 4::] = LUL
    test_im[0, 5, 5::] = LUL; test_im[1, 5, 5::] = LUL
    test_im[0, 6, 5::] = LUL; test_im[1, 6, 5::] = LUL
    test_im[0, 7, 5::] = LUL; test_im[1, 7, 5::] = LUL
    
    points = vtk.vtkPoints()
    points.InsertNextPoint([0, 0, 1])
    points.InsertNextPoint([0, 1, 1])
    points.InsertNextPoint([1, 0, 1])
    points.InsertNextPoint([1, 1, 1.01])

    irad_arr = vtk.vtkFloatArray()
    irad_arr.InsertNextTuple1(1)
    irad_arr.SetName('irad')
    
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.GetFieldData().AddArray(irad_arr)
    
    df = completeness_phenos.execute(test_im, np.array([0, 0, 0]),
        np.array([1, 1, 1]), 'foo', lop_poly=poly, completeness_type='domain')

    assert df.Completeness.values[0] == 0.375, \
        "Completeness measure should equal 0.375"
