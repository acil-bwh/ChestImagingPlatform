import pandas as pd

#import cip_python.utils as utils
from cip_python.utils import ReadGeometryTopologyPointWriteDataFrame
import cip_python.common as common
from optparse import OptionParser

import vtk

class ReadGeometryTopologyPointWritePolyData():
    """
        Reads a geometry topology data, an encapsuales in a vtkPolyData the points from a given Region/Type
        
        Parameters
        ----------
        in_file_name: string
        input xml filename containing the geometrytopology data
        
        ------
        """

    def __init__(self,in_file_name):
        self.in_file_name = in_file_name

    def execute(self,region_name_list=None,type_name_list=None):
        
        c=common.ChestConventions()
        
        reader=ReadGeometryTopologyPointWriteDataFrame(self.in_file_name)
        reader.execute()
        
        df=reader.df_

        points = vtk.vtkPoints()
        regionTypeArr = vtk.vtkUnsignedShortArray()
        regionTypeArr.SetName('ChestRegionChestType')
        if region_name_list == None:
            region_name_list=list(df['Region'].unique())
        if type_name_list == None:
            type_name_list=list(df['Type'].unique())

        for rr in region_name_list:
            for tt in type_name_list:
                xml_points = df[(df['Region'] == rr) & (df['Type'] == tt)]
                rr_val = c.GetChestRegionValueFromName(rr)
                tt_val = c.GetChestTypeValueFromName(tt)
                for pp in xml_points.iterrows():
                    points.InsertNextPoint(float(pp[1]['X point']), float(pp[1]['Y point']), float(pp[1]['Z point']))
                    regionTypeArr.InsertNextValue(c.GetValueFromChestRegionAndType(rr_val, tt_val))

        points_pd = vtk.vtkPolyData()
        points_pd.SetPoints(points)
        points_pd.GetPointData().AddArray(regionTypeArr)
        return points_pd

if __name__ == "__main__":
    desc = """Generates a vtk PolyData with region and type points from
        geometry_topology data"""

    parser = OptionParser(description=desc)
    parser.add_option('-i',help='Input xml file', dest='in_xml', \
        metavar='<string>',default=None)
    parser.add_option('-r',
        help='Chest regions. Should be specified as a \
        common-separated list of string values indicating the \
        desired regions over which to extract points. \
        If regions are not specified, all the regions will be extracted. \
        E.g. LeftLung,RightLung would be a valid input.',
        dest='chest_regions', metavar='<string>', default=None)
    parser.add_option('-t',
        help='Chest types. Should be specified as a \
        common-separated list of string values indicating the \
        desired types over which to extract points. \
        If types are not specified, all the types will be extracted. \
        E.g.: Vessel,NormalParenchyma would be a valid input.',
        dest='chest_types', metavar='<string>', default=None)
    parser.add_option('-o',
        help='Output vtkPolyData file (.vtk)', \
        dest='out_polydata', metavar='<string>',default=None)

    (options, args) = parser.parse_args()
           
    regions = None
    if options.chest_regions is not None:
        regions = options.chest_regions.split(',')
    types = None
    if options.chest_types is not None:
        types = options.chest_types.split(',')


    filter = ReadGeometryTopologyPointWritePolyData(options.in_xml)
    out_polydata = filter.execute(regions,types)
                                                  
    writer=vtk.vtkPolyDataWriter()
    writer.SetInputData(out_polydata)
    writer.SetFileName(options.out_polydata)
    writer.Write()


