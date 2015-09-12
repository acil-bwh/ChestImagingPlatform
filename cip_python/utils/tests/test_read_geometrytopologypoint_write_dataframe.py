from cip_python.utils.read_geometrytopologypoint_write_dataframe import  *
import pdb
import os.path
import pandas as pd
from pandas.util.testing import assert_frame_equal

def test_execute():
    
    # convert to csv (we get a pandas dataframe)
    this_dir = os.path.dirname(os.path.realpath(__file__))
    in_xml = this_dir + \
        '/../../../Testing/Data/Input/simple_regionAndTypePoints.xml'
    in_xml = this_dir + \
        '/../../../Testing/Data/Input/geometryTopologyData.xml'
    
    my_csv_writer = ReadGeometryTopologyPointWriteDataFrame(in_file_name = \
        in_xml)     
    my_csv_writer.execute()  

    # compare to expected pandas dataframe
    
    cols = ['Region','Type', 'X point', 'Y point', 'Z point']
    ref_df = pd.DataFrame(columns=cols)
    tmp = dict()
    tmp['Region'] = 'RightLung'
    tmp['Type'] = 'GroundGlass'
    tmp['X point'] = 2
    tmp['Y point'] = 3.5
    tmp['Z point'] = 3
    ref_df= ref_df.append(tmp, ignore_index=True)
    tmp['X point'] = 2
    tmp['Y point'] = 3.0
    tmp['Z point'] = 3
    ref_df= ref_df.append(tmp, ignore_index=True)
    assert_frame_equal(my_csv_writer.df_, ref_df)   
        