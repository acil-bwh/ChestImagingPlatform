import os, sys
from lxml import etree
from cip_python.utils.geometry_topology_data import *


this_dir = os.path.dirname(os.path.realpath(__file__))     # Directory where this file is contained
xml_file = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "Testing", "Data", "Input", "geometryTopologyData.xml"))
xsd_file = os.path.abspath(os.path.join(this_dir, "..", "..", "..", "Resources", "Schemas", "GeometryTopologyData.xsd"))

def test_geometry_topology_data_schema():
    """ Validate the current sample xml file (geometryTopologyData-sample.xml) with the current schema
    """
    # Read xml
    with open(xml_file, 'r+b') as f:
        xml = f.read()

    # Validate schema with lxml
    with open(xsd_file, 'r+b') as f:
        xsd = f.read()
    schema = etree.XMLSchema(etree.XML(xsd))
    xmlparser = etree.XMLParser(schema=schema)
    etree.fromstring(xml, xmlparser)


def test_geometry_topology_data_write_read():
    """ Create a GeometryTopology object that must be equal to the one in xml_file.
    It also validates the xml schema against the xsd file
    """
    # Create a new object from scratch
    g = GeometryTopologyData()
    g.num_dimensions = 3
    g.coordinate_system = g.RAS
    g.lps_to_ijk_transformation_matrix = [[-1.9, 0, 0, 250], [0, -1.9, 0, 510], [0, 0, 2, 724], [0, 0, 0, 1]]

    g.add_point(Point(2, 5, 1, [2, 3.5, 3], description="My desc", format_="%f"))
    g.add_point(Point(2, 5, 1, coordinate=[2, 3.5, 3], format_="%i"))
    g.add_bounding_box(BoundingBox(2, 5, 1, start=[2, 3.5, 3], size=[1, 1, 4], format_="%i"))
    g.add_bounding_box(BoundingBox(2, 5, 1, start=[2, 3.5, 3], size=[1, 1, 3], format_="%f"))

    # Get xml representation for the object
    xml = g.to_xml()

    # Compare XML output with the example file
    with open(xml_file, 'r+b') as f:
        expectedOutput = f.read()
    assert xml == expectedOutput, "XML generated: " + xml

    # Validate schema with lxml
    with open(xsd_file, 'r+b') as f:
        xsd = f.read()
    schema = etree.XMLSchema(etree.XML(xsd))
    xmlparser = etree.XMLParser(schema=schema)
    etree.fromstring(xml, xmlparser)
