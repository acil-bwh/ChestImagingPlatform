"""
Classes that represent a collection of points/structures that will define a labelmap or similar for image analysis purposes.
Currently the parent object is GeometryTopologyData, that can contain objects of type Point and/or BoundingBox.
The structure of the object is defined in the GeometryTopologyData.xsd schema.
Created on Apr 6, 2015

@author: Jorge Onieva
"""

import xml.etree.ElementTree as et

class GeometryTopologyData:
    # Coordinate System Constants
    UNKNOWN = 0
    IJK = 1
    RAS = 2
    LPS = 3

    __num_dimensions__ = 0
    @property
    def num_dimensions(self):
        """ Number of dimensions (generally 3)"""
        if self.__num_dimensions__ == 0:
            # Try to get the number of dimensions from the first point or bounding box
            if len(self.points) > 0:
                self.__num_dimensions__ = len(self.points[0].coordinate)
            elif len(self.bounding_boxes) > 0:
                self.__num_dimensions__ = len(self.bounding_boxes[0].start)
        return self.__num_dimensions__
    @num_dimensions.setter
    def num_dimensions(self, value):
        self.__num_dimensions__ = value


    def __init__(self):
        self.__num_dimensions__ = 0
        self.coordinate_system = self.UNKNOWN
        self.lps_to_ijk_transformation_matrix = None    # 4x4 transformation matrix to go from LPS to IJK (in the shape of a 4x4 list)

        self.points = []    # List of Point objects
        self.bounding_boxes = []    # List of BoundingBox objects

    def add_point(self, point):
        """ Add a new Point to the structure
        :param point: Point object """
        self.points.append(point)

    def add_bounding_box(self, bounding_box):
        """ Add a new BoundingBox to the structure
        :param bounding_box: BoundingBox object
        :return:
        """
        self.bounding_boxes.append(bounding_box)

    def to_xml(self):
        """ Generate the XML string representation of this object.
        It doesn't use any special python module to keep compatibility with Slicer """
        output = '<?xml version="1.0" encoding="utf8"?><GeometryTopologyData>'
        if self.num_dimensions != 0:
            output += ('<NumDimensions>%i</NumDimensions>' % self.num_dimensions)

        output += ('<CoordinateSystem>%s</CoordinateSystem>' % self.coordinate_system_to_str(self.coordinate_system))

        if self.lps_to_ijk_transformation_matrix is not None:
            output += self.write_transformation_matrix(self.lps_to_ijk_transformation_matrix)

        # Concatenate points
        points = "".join(map(lambda i:i.to_xml(), self.points))
        # Concatenate bounding boxes
        bounding_boxes = "".join(map(lambda i:i.to_xml(), self.bounding_boxes))

        return output + points + bounding_boxes + "</GeometryTopologyData>"

    @staticmethod
    def from_xml(xml):
        """ Build a GeometryTopologyData object from a xml string.
        All the coordinates will be float.
        remark: It uses the ElementTree instead of lxml module to be compatible with Slicer
        :param xml: xml string
        :return: new GeometryTopologyData object
        """
        root = et.fromstring(xml)
        geometry_topology = GeometryTopologyData()

        # NumDimensions
        s = root.find("NumDimensions")
        if s is not None:
            geometry_topology.__num_dimensions__ = int(s.text)

        # Coordinate System
        s = root.find("CoordinateSystem")
        if s is not None:
            geometry_topology.coordinate_system = geometry_topology.coordinate_system_from_str(s.text)

        geometry_topology.lps_to_ijk_transformation_matrix = geometry_topology.read_transformation_matrix(root)

        # Points
        for point in root.findall("Point"):
            geometry_topology.add_point(Point.from_xml_node(point))

            # coordinates = []
            # for coord in point.findall("Coordinate/value"):
            #     coordinates.append(float(coord.text))
            # chest_region = int(point.find("ChestRegion").text)
            # chest_type = int(point.find("ChestType").text)
            #
            # # Description
            # desc = point.find("Description")
            # if desc is not None:
            #     desc = desc.text
            #
            # geometry_topology.add_point(Point(coordinates, chest_region, chest_type, description=desc))

        # BoundingBoxes
        for bb in root.findall("BoundingBox"):
            geometry_topology.add_bounding_box(BoundingBox.from_xml_node(point))
            # coordinates_start = []
            # for coord in bb.findall("Start/value"):
            #     coordinates_start.append(float(coord.text))
            # coordinates_size = []
            # for coord in bb.findall("Size/value"):
            #     coordinates_size.append(float(coord.text))
            # chest_region = int(bb.find("ChestRegion").text)
            # chest_type = int(bb.find("ChestType").text)
            #
            # # Description
            # desc = bb.find("Description")
            # if desc is not None:
            #     desc = desc.text
            #
            # geometry_topology.add_bounding_box(BoundingBox(coordinates_start, coordinates_size, chest_region, chest_type, description=desc))

        return geometry_topology

    @staticmethod
    def to_xml_vector(array, format_="%f"):
        """ Get the xml representation of a vector of coordinates (<value>elem1</value>, <value>elem2</value>...)
        :param array: vector of values
        :return: xml representation of the vector (<value>elem1</value>, <value>elem2</value>...)
        """
        output = ''
        for i in array:
            output = ("%s<value>" + format_ + "</value>") % (output, i)
        return output

    @staticmethod
    def coordinate_system_from_str(value_str):
        """ Get one of the possible coordinate systems allowed from its string representation
        :param value_str: "IJK", "RAS", "LPS"...
        :return: one the allowed coordinates systems
        """
        if value_str is not None:
            if value_str == "IJK": return GeometryTopologyData.IJK
            elif value_str == "RAS": return GeometryTopologyData.RAS
            elif value_str == "LPS": return GeometryTopologyData.LPS
            else: return GeometryTopologyData.UNKNOWN
        else:
            return GeometryTopologyData.UNKNOWN

    @staticmethod
    def coordinate_system_to_str(value_int):
        """ Get the string representation of one of the coordinates systems
        :param value_int: GeometryTopologyData.IJK, GeometryTopologyData.RAS, GeometryTopologyData.LPS...
        :return: string representing the coordinate system ("IJK", "RAS", "LPS"...)
        """
        if value_int == GeometryTopologyData.IJK: return "IJK"
        elif value_int == GeometryTopologyData.RAS: return "RAS"
        elif value_int == GeometryTopologyData.LPS: return "LPS"
        return "UNKNOWN"

    def read_transformation_matrix(self, root_xml):
        """ Read a 16 elems vector in the xml and return a 4x4 list (or None if node not found)
        :param root_xml: xml root node
        :return: 4x4 list or None
        """
        # Try to find the node first
        node = root_xml.find("LPStoIJKTransformationMatrix")
        if node is None:
            return None
        m = []
        temp = []
        for coord in node.findall("value"):
            temp.append(float(coord.text))

        # Convert to a 4x4 list
        for i in range (4):
            m.append([temp[i*4], temp[i*4+1], temp[i*4+2], temp[i*4+3]])
        return m

    def write_transformation_matrix(self, matrix):
        """ Generate an xml text for a 4x4 transformation matrix
        :param matrix: 4x4 list
        :return: xml string (LPStoIJKTransformationMatrix complete node)
        """
        # Flatten the list
        s = ""
        for item in (item for sublist in matrix for item in sublist):
            s += ("<value>%f</value>" % item)
        return "<LPStoIJKTransformationMatrix>%s</LPStoIJKTransformationMatrix>" % s


class Point:
    def __init__(self, chest_region, chest_type, feature_type, coordinate, description=None, format_="%f"):
        """
        :param coordinate: Vector of numeric coordinates
        :param chest_region: chestRegion Id
        :param chest_type: chestType Id
        :param feature_type: feature type Id (artifacts and others)
        :param description: optional description of the content the element
        :param format_: Default format to print the xml output coordinate values (also acceptable: %i for integers or customized)
        :return:
        """
        self.coordinate = coordinate
        self.chest_region = chest_region
        self.chest_type = chest_type
        self.feature_type = feature_type
        self.description = description
        self.format = format_

    @staticmethod
    def from_xml_node(xml_point_node):
        """ Return a new instance of a Point object from xml "Point" element
        :param xml_point_node: xml Point element coming from a "find" instruction
        :return: new instance of Point
        """
        print("Point: ", xml_point_node)
        coordinates = []
        for coord in xml_point_node.findall("Coordinate/value"):
            coordinates.append(float(coord.text))
        chest_region = int(xml_point_node.find("ChestRegion").text)
        chest_type = int(xml_point_node.find("ChestType").text)
        feature_type = int(xml_point_node.find("ImageFeature").text)

        # Description
        desc = xml_point_node.find("Description")
        if desc is not None:
            desc = desc.text

        return Point(chest_region, chest_type, feature_type, coordinates, description=desc)

    def to_xml(self):
        """ Get the xml string representation of the point
        :return: xml string representation of the point
        """
        coords = GeometryTopologyData.to_xml_vector(self.coordinate, self.format)
        description_str = ''
        if self.description is not None:
            description_str = '<Description>%s</Description>' % self.description

        return '<Point><ChestRegion>%i</ChestRegion><ChestType>%i</ChestType><ImageFeature>%i</ImageFeature>%s<Coordinate>%s</Coordinate></Point>' % \
            (self.chest_region, self.chest_type, self.feature_type, description_str, coords)


class BoundingBox:
    def __init__(self, chest_region, chest_type, feature_type, start, size, description=None, format_="%f"):
        """
        :param start: vector of coordinates for the starting point of the Bounding Box
        :param size: vector that contains the size of the bounding box
        :param chest_region: chestRegion Id
        :param chest_type: chestType Id
        :param feature_type: feature type Id (artifacts and others)
        :param description: optional description of the content the element
        :param format_: Default format to print the xml output coordinate values (also acceptable: %i for integers or customized)
        :return:
        """
        self.start = start
        self.size = size
        self.chest_region = chest_region
        self.chest_type = chest_type
        self.feature_type = feature_type
        self.description = description
        self.format = format_       # Default format to print the xml output coordinate values (also acceptable: %i or customized)


    @staticmethod
    def from_xml_node(xml_bounding_box_node):
        """ Return a new instance of a Point object from xml "BoundingBox" element
        :param xml_bounding_box_node: xml BoundingBox element coming from a "find" instruction
        :return: new instance of BoundingBox
        """
        coordinates_start = []
        for coord in xml_bounding_box_node.findall("Start/value"):
            coordinates_start.append(float(coord.text))
        coordinates_size = []
        for coord in xml_bounding_box_node.findall("Size/value"):
            coordinates_size.append(float(coord.text))
        chest_region = int(xml_bounding_box_node.find("ChestRegion").text)
        chest_type = int(xml_bounding_box_node.find("ChestType").text)
        feature_type = int(xml_bounding_box_node.find("ImageFeature").text)

        # Description
        desc = xml_bounding_box_node.find("Description")
        if desc is not None:
            desc = desc.text

        return BoundingBox(chest_region, chest_type, feature_type, coordinates_start, coordinates_size, description=desc)


    def to_xml(self):
        """ Get the xml string representation of the bounding box
        :return: xml string representation of the bounding box
        """
        start_str = GeometryTopologyData.to_xml_vector(self.start, self.format)
        size_str = GeometryTopologyData.to_xml_vector(self.size, self.format)
        description_str = ''
        if self.description is not None:
            description_str = '<Description>%s</Description>' % self.description
        return '<BoundingBox><ChestRegion>%i</ChestRegion><ChestType>%i</ChestType><ImageFeature>%i</ImageFeature>%s<Start>%s</Start><Size>%s</Size></BoundingBox>' % \
            (self.chest_region, self.chest_type, self.feature_type, description_str, start_str, size_str)
            



