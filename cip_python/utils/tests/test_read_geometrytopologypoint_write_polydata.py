import pandas as pd
from pandas.util.testing import assert_frame_equal
import vtk
import numpy as np
from cip_python.utils import ReadGeometryTopologyPointWritePolyData
from cip_python.common import Paths

def test_airway_data():
    # convert to csv (we get a pandas dataframe)
    in_xml = Paths.testing_file_path('geometryTopologyDataLarge.xml')

    filter = ReadGeometryTopologyPointWritePolyData(in_xml)
    out_poly=filter.execute()

    regions=['RightSuperiorLobe','RightInferiorLobe']
    types=['AirwayGeneration3','AirwayGeneration4']

    out_pd = filter.execute(regions,types)
    
    # compare poly data to reference dataset
    ref_reader = vtk.vtkPolyDataReader()
    ref_reader.SetFileName(Paths.testing_file_path('geometryTopologyDataLarge_airways.vtk'))
    ref_reader.Update()
    
    ref_pd=ref_reader.GetOutput()
    
    assert ref_pd.GetNumberOfPoints() == out_pd.GetNumberOfPoints()
    
    # Check points
    for pp in xrange(ref_pd.GetNumberOfPoints()):
        p_1 = ref_pd.GetPoints().GetPoint(pp)
        p_2 = out_pd.GetPoints().GetPoint(pp)
        print p_1
        print p_2
        assert (np.isclose(p_1[0],p_2[0]) and np.isclose(p_1[1],p_2[1]) and np.isclose(p_1[2],p_2[2])), 'Error not exact'
    
    # Check RegionType Field
    for pp in xrange(ref_pd.GetNumberOfPoints()):
        val_1 = ref_pd.GetPointData().GetArray('ChestRegionChestType').GetValue(pp)
        val_2 = out_pd.GetPointData().GetArray('ChestRegionChestType').GetValue(pp)
        assert val_1 == val_2, 'Error'

