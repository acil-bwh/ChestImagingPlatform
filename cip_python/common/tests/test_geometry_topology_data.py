import os
import tempfile
from lxml import etree
from cip_python.common import *

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
    g.coordinate_system = g.RAS
    g.lps_to_ijk_transformation_matrix = [[-1.9, 0, 0, 250], [0, -1.9, 0, 510], [0, 0, 2, 724], [0, 0, 0, 1]]
    g.spacing = (0.7, 0.7, 0.5)
    g.origin = (180.0, 180.0, -700.5)
    g.dimensions = (512, 512, 600)

    p1 = Point(ChestRegion.RIGHTLUNG, ChestType.GROUNDGLASS, ImageFeature.CTARTIFACT, [2, 3.5, 3], description="My desc")
    p1.__id__ = 1
    p1.timestamp = "2015-10-21 04:00:00"
    p1.user_name = "mcfly"
    p1.machine_name = "DELOREAN"
    g.add_point(p1, fill_auto_fields=False)
    p2 = Point(ChestRegion.LEFTLUNG, ChestType.AIRWAY, ImageFeature.UNDEFINEDFEATURE, [2.0, 1.5, 3.75])
    p2.__id__ = 2
    p2.timestamp = p1.timestamp
    p2.user_name = p1.user_name
    p2.machine_name = p1.machine_name
    g.add_point(p2, fill_auto_fields=False)
    bb1 = BoundingBox(ChestRegion.LEFTLUNG, ChestType.AIRWAY, ImageFeature.UNDEFINEDFEATURE, [2, 3.5, 3], [1, 1, 4])
    bb1.__id__ = 3
    bb1.timestamp = p1.timestamp
    bb1.user_name = p1.user_name
    bb1.machine_name = p1.machine_name
    g.add_bounding_box(bb1, fill_auto_fields=False)
    bb2 = BoundingBox(ChestRegion.RIGHTLUNG, ChestType.GROUNDGLASS, ImageFeature.CTARTIFACT, [2, 3.5, 3], [2.0, 2, 5], description="My desc")
    bb2.__id__ = 4
    bb2.timestamp = p1.timestamp
    bb2.user_name = p1.user_name
    bb2.machine_name = p1.machine_name
    g.add_bounding_box(bb2, fill_auto_fields=False)

    # Write the object to a xml file
    output_file = os.path.join(tempfile.gettempdir(), "geom.xml")
    g.to_xml_file(output_file)
    print ("Temp file created: {}".format(output_file))

    # Compare XML output with the example file
    with open(xml_file, 'r+b') as f:
        expected_output = f.read()

    with open(output_file, 'r+b') as f:
        generated_output = f.read()

    # Remove \r to avoid platform compatibility issues
    expected_output = expected_output.replace('\r', '')
    generated_output = generated_output.replace('\r', '')

    assert generated_output == expected_output

    # Remove temp file
    os.remove(output_file)

    # Validate schema with lxml
    with open(xsd_file, 'r+b') as f:
        xsd = f.read()
    schema = etree.XMLSchema(etree.XML(xsd))
    xmlparser = etree.XMLParser(schema=schema)
    etree.fromstring(generated_output, xmlparser)

    # Make sure that the seed is set to a right value
    g.update_seed()
    assert g.seed_id == 5, "Seed in the object should be 5, while the current value is {}".format(g.seed_id)

