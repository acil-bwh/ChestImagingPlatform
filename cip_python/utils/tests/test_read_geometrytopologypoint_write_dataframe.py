import pandas as pd
from pandas.util.testing import assert_frame_equal

from cip_python.utils import ReadGeometryTopologyPointWriteDataFrame
from cip_python.common import Paths


def test_execute():
    # convert to csv (we get a pandas dataframe)
    in_xml = Paths.testing_file_path('geometryTopologyData.xml')

    my_csv_writer = ReadGeometryTopologyPointWriteDataFrame(in_file_name=in_xml)
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
    tmp['Region'] = 'LeftLung'
    tmp['Type'] = 'Airway'
    tmp['X point'] = 2
    tmp['Y point'] = 1.5
    tmp['Z point'] = 3.75
    ref_df= ref_df.append(tmp, ignore_index=True)
    assert_frame_equal(my_csv_writer.df_, ref_df, check_dtype=False)
