from cip_python.common import *
from lxml import etree

xml_file = Paths.testing_file_path('geometryTopologyData.xml')
xsd_file = Paths.resources_file_path('Schemas/GeometryTopologyData.xsd')

def test_geometry_topology_data_schema():
    """ Validate the current sample xml file (Testing/Data/Input/geometryTopologyData.xml) with the current schema
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

    p1 = Point(2, 5, 1, [2, 3.5, 3], description="My desc", format_="%f")
    p1.__id__ = 1
    p1.timestamp = "2015-10-21 04:00:00"
    p1.user_name = "mcfly"
    p1.machine_name = "DELOREAN"
    g.add_point(p1, fill_auto_fields=False)
    p2 = Point(2, 5, 1, coordinate=[2, 3.5, 3], format_="%i")
    p2.__id__ = 2
    p2.timestamp = p1.timestamp
    p2.user_name = p1.user_name
    p2.machine_name = p1.machine_name
    g.add_point(p2, fill_auto_fields=False)
    bb1 = BoundingBox(2, 5, 1, start=[2, 3.5, 3], size=[1, 1, 4], format_="%i")
    bb1.__id__ = 3
    bb1.timestamp = p1.timestamp
    bb1.user_name = p1.user_name
    bb1.machine_name = p1.machine_name
    g.add_bounding_box(bb1, fill_auto_fields=False)
    bb2 = BoundingBox(2, 5, 1, start=[2, 3.5, 3], size=[1, 1, 3], format_="%f")
    bb2.__id__ = 4
    bb2.timestamp = p1.timestamp
    bb2.user_name = p1.user_name
    bb2.machine_name = p1.machine_name
    g.add_bounding_box(bb2, fill_auto_fields=False)

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
