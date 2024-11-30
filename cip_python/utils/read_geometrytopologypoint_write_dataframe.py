import os
from optparse import OptionParser

import pandas as pd
import cip_python.common as common
from lxml import etree


class ReadGeometryTopologyPointWriteDataFrame:
    """
    Class that reads geometry topology data and returns a dataframe with 
    region and type points.
    
    Parameters
    ----------
    in_file_name: string
        input xml filename containing the geometrytopology data
        
    df_: Pandas dataframe
        dataframe containing the geometrytopology data after being read
    ------
    """
    def __init__(self, in_file_name):
        self.in_file_name = in_file_name
        cols = ['Region','Type', 'X point', 'Y point', 'Z point'] #, 'FeatureType', 'Description', 'Timestamp', 'UserName', 'MachineName']
        self.df_ = pd.DataFrame(columns=cols)
        
    def execute(self):
        with open(self.in_file_name, 'r+b') as f:
            xml = f.read()
            
        # Validate schema with lxml
        this_dir = os.path.dirname(os.path.realpath(__file__))
        xsd_file = os.path.abspath(os.path.join(this_dir, "..", "..", \
            "Resources", "Schemas", "GeometryTopologyData.xsd"))

        with open(xsd_file, 'r+b') as f:
            xsd = f.read()
        schema = etree.XMLSchema(etree.XML(xsd))
        xmlparser = etree.XMLParser(schema=schema)
        etree.fromstring(xml, xmlparser)
            
        my_geometry_data = common.GeometryTopologyData.from_xml(xml)

        c = common.ChestConventions()
        for the_point in my_geometry_data.points:
            tmp = dict()
            tmp['Region'] = c.GetChestRegionName(the_point.chest_region)
            tmp['Type'] = c.GetChestTypeName(the_point.chest_type) 
            tmp['X point'] = the_point.coordinate[0]
            tmp['Y point'] = the_point.coordinate[1]
            tmp['Z point'] = the_point.coordinate[2]
            # tmp['FeatureType'] = c.GetImageFeatureName(the_point.feature_type) if the_point.feature_type else None
            # tmp['Description'] = the_point.description
            # tmp['Timestamp'] = the_point.timestamp
            # tmp['UserName'] = the_point.user_name
            # tmp['MachineName'] = the_point.machine_name

            #feature_type = c.GetImageFeatureName(the_point.feature_type)
            #print(feature_type)
            self.df_ = self.df_.append(tmp, ignore_index=True)

if __name__ == "__main__":
    desc = """Generates a csv file with region and type points from 
        geometry_topology data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--xml',
                      help='Input xml file', dest='in_xml', \
                      metavar='<string>', default=None)
    parser.add_option('--out_csv',
                      help='Output csv file.', 
                      dest='out_csv', metavar='<string>', default=None)
    (options, args) = parser.parse_args()
                      
    my_csv_writer = ReadGeometryTopologyPointWriteDataFrame(in_file_name = \
        options.in_xml)     
    my_csv_writer.execute()             
    my_csv_writer.df_.to_csv(options.out_csv, index=False)                  