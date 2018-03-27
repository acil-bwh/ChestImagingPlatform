"""
Classes that represent a collection of points/structures that will define a labelmap or similar for image analysis purposes.
Currently the parent object is GeometryTopologyData, that can contain objects of type Point and/or BoundingBox.
The structure of the object is defined in the GeometryTopologyData.xsd schema.
Created on Apr 6, 2015

@author: Jorge Onieva
"""

import xml.etree.ElementTree as et

import os
import platform
import time
import numpy as np
import warnings

class GeometryTopologyData(object):
    # Coordinate System Constants
    UNKNOWN = 0
    IJK = 1
    RAS = 2
    LPS = 3

    def __init__(self):
        self.coordinate_system = self.UNKNOWN
        self.lps_to_ijk_transformation_matrix = None    # Transformation matrix to go from LPS to IJK (in the shape of a 4x4 list)
        self.__lps_to_ijk_transformation_matrix_array__ = None  # Same matrix in a numpy array

        self.origin = None      # Volume origin
        self.spacing = None     # Volume spacing
        self.dimensions = None  # Volume Dimensions

        self.points = []    # List of Point objects
        self.bounding_boxes = []    # List of BoundingBox objects

        self.__seed_id__ = 1    # Seed. The structures added with "add_point", etc. will have an id = seed_id + 1

    @property
    def seed_id(self):
        return self.__seed_id__

    @seed_id.setter
    def seed_id(self, value):
        warnings.warn("This property should not be set manually. Use with caution")
        self.__seed_id__ = value

    @property
    def lps_to_ijk_transformation_matrix_array(self):
        """ LPS_IJK transformation matrix in a numpy format
        """
        if self.lps_to_ijk_transformation_matrix is None:
            return None
        if self.__lps_to_ijk_transformation_matrix_array__ is None:
            self.__lps_to_ijk_transformation_matrix_array__ = np.array(self.lps_to_ijk_transformation_matrix,
                                                                       dtype=np.float)
        return self.__lps_to_ijk_transformation_matrix_array__

    def __str__(self):
        """
        Print a nicely formatted XML with the current content of the object
        """
        return self.to_xml(pretty_print=True)

    def add_point(self, point, fill_auto_fields=True, timestamp=None):
        """ Add a new Point to the structure
        :param point: Point object
        :param fill_auto_fields: fill automatically UserName, MachineName, etc.
        :param timestamp: optional timestamp to be set in the object
        """
        self.points.append(point)
        if fill_auto_fields:
            self.fill_auto_fields(point)
        if timestamp:
            point.timestamp = timestamp

    def add_bounding_box(self, bounding_box, fill_auto_fields=True, timestamp=None):
        """ Add a new BoundingBox to the structure
        :param bounding_box: BoundingBox object
        :param fill_auto_fields: fill automatically UserName, MachineName, etc.
        :param timestamp: optional timestamp to be set in the object
        """
        self.bounding_boxes.append(bounding_box)
        if fill_auto_fields:
            self.fill_auto_fields(bounding_box)
        if timestamp:
            bounding_box.timestamp = timestamp

    def fill_auto_fields(self, structure):
        """ Fill "auto" fields like timestamp, username, etc, unless there is already a specified value
        The id will be id_seed + 1
        @param structure: object whose fields will be filled
        """
        if structure.__id__ == 0:
            structure.__id__ = self.__seed_id__
            self.__seed_id__ += 1
        if not structure.timestamp:
            structure.timestamp = GeometryTopologyData.get_timestamp()
        if not structure.user_name:
            structure.user_name = os.path.split(os.path.expanduser('~'))[-1]
        if not structure.machine_name:
            structure.machine_name = platform.node()

    def update_seed(self):
        """
        Update the id_seed field to the maximum id found + 1
        """
        id = 0
        for p in self.points:
            id = max(id, p.id)
        for bb in self.bounding_boxes:
            id = max(id, bb.id)
        self.__seed_id__ = id + 1

    @staticmethod
    def get_timestamp():
        """ Get a timestamp of the current date in the preferred format
        @return:
        """
        return time.strftime('%Y-%m-%d %H:%M:%S')

    def to_xml(self, pretty_print=False):
        """
        Generate the XML string representation of this object.
        It doesn't use any special python module by default to keep compatibility with Slicer
        Args:
            pretty_print: when True and lxml is available, print the xml in a nice format
        Returns:
            XML string representation of the object
        """
        header = '<?xml version="1.0" encoding="UTF-8"?>'

        output = header + "<GeometryTopologyData>"

        output += ("<CoordinateSystem>%s</CoordinateSystem>" % self.__coordinate_system_to_str__(self.coordinate_system))

        if self.lps_to_ijk_transformation_matrix is not None:
            output += self.__write_transformation_matrix__(self.lps_to_ijk_transformation_matrix)

        if self.spacing is not None:
            output += "<Spacing>{}</Spacing>".format(GeometryTopologyData.__to_xml_vector__(self.spacing))
        if self.origin is not None:
            output += "<Origin>{}</Origin>".format(GeometryTopologyData.__to_xml_vector__(self.origin))
        if self.dimensions is not None:
            output += "<Dimensions>{}</Dimensions>".format(GeometryTopologyData.__to_xml_vector__(self.dimensions))

        # Concatenate points (sort first)
        self.points.sort(key=lambda p: p.__id__)
        points = "".join(map(lambda i:i.to_xml(), self.points))
        # Concatenate bounding boxes
        bounding_boxes = "".join(map(lambda i:i.to_xml(), self.bounding_boxes))

        s = output + points + bounding_boxes + "</GeometryTopologyData>"
        if pretty_print:
            try:
                import lxml.etree as etree
                s = str(s)  # Avoid Unicode
                x = etree.fromstring(s)
                s = etree.tostring(x, pretty_print=True, xml_declaration="1.0", encoding="UTF-8")
                # Replace ' just to be sure that the format remains exactly the same
                s = s.replace("<?xml version='1.0' encoding='UTF-8'?>", header)
            except:
                print ("lxml not found. Nice print not available")
        return s

    def to_xml_file(self, xml_file_path, pretty_print=True):
        """
        Save this object to an xml file
        Args:
            xml_file_path: file path
            pretty_print: write the xml in a nice format (requires lxml)
        """
        s = self.to_xml(pretty_print=pretty_print)
        with open(xml_file_path, "w+b") as f:
            f.write(s)

    @staticmethod
    def from_xml_file(xml_file_path):
        """ Get a GeometryTopologyObject from a file
        @param xml_file_path: file path
        @return: GeometryTopologyData object
        """
        with open(xml_file_path, 'r+b') as f:
            xml = f.read()
            return GeometryTopologyData.from_xml(xml)

    @staticmethod
    def from_xml(xml):
        """ Build a GeometryTopologyData object from a xml string.
        All the coordinates will be float.
        remark: Use the ElementTree instead of lxml module to be compatible with Slicer
        :param xml: xml string
        :return: new GeometryTopologyData object
        """
        root = et.fromstring(xml)
        geometry_topology = GeometryTopologyData()

        # NumDimensions. DEPRECATED
        # node = root.find("NumDimensions")
        # if node is not None:
        #     geometry_topology.__num_dimensions__ = int(node.text)

        # Coordinate System
        node = root.find("CoordinateSystem")
        if node is not None:
            geometry_topology.coordinate_system = geometry_topology.__coordinate_system_from_str__(node.text)

        geometry_topology.lps_to_ijk_transformation_matrix = geometry_topology.__read_transformation_matrix__(root)

        node = root.find("Spacing")
        if node is not None:
            val = []
            for node_val in node.findall("value"):
                val.append(float(node_val.text))
            geometry_topology.spacing = np.array(val)

        node = root.find("Origin")
        if node is not None:
            val = []
            for node_val in node.findall("value"):
                val.append(float(node_val.text))
            geometry_topology.origin = np.array(val)

        node = root.find("Dimensions")
        if node is not None:
            val = []
            for node_val in node.findall("value"):
                val.append(float(node_val.text))
            geometry_topology.origin = np.array(val)

        # Points
        for xml_point_node in root.findall("Point"):
            point = Point.from_xml_node(xml_point_node)
            geometry_topology.add_point(point, fill_auto_fields=False)

        # BoundingBoxes
        for xml_bb_node in root.findall("BoundingBox"):
            bb = BoundingBox.from_xml_node(xml_bb_node)
            geometry_topology.add_bounding_box(BoundingBox.from_xml_node(xml_bb_node), fill_auto_fields=False)

        # Set the new seed so that every point (or bounding box) added with "add_point" has a bigger id
        geometry_topology.update_seed()

        return geometry_topology

    def get_hashtable(self):
        """
        Return a "hashtable" that will be a dictionary of hash:structure for every point or
        bounding box present in the structure
        """
        hash = {}
        for p in self.points:
            hash[p.get_hash()] = p
        for bb in self.bounding_boxes:
            hash[bb.get_hash()] = bb
        return hash

    def convert_coordinates_to_array(self, type_=np.float32):
        """
        Convert the coordinates of all the Points/Bounding_Boxes to numpy arrays of the specific type (default: float32)
        Args:
            type_: type for the conversion (default: float32)
        """
        for p in self.points:
            p.convert_to_array(type_)
        for bb in self.bounding_boxes:
            bb.convert_to_array(type_)

    def coordinate_system_str(self):
        """
        Return the coordinate system in text ("LPS", "RAS", "IJK", "UNKNOWN")
        Returns: string
        """
        if self.coordinate_system == self.IJK:
            return "IJK"
        if self.coordinate_system == self.RAS:
            return "RAS"
        if self.coordinate_system == self.LPS:
            return "LPS"
        return "UNKNOWN"


    def export_to_dataframe(self):
        import pandas as pd
        from cip_python.common import ChestConventions
        if len(self.points) > 0 and len(self.bounding_boxes) > 0:
            raise NotImplementedError("This function can be used only for points or bounding boxes. This object contains both")

        columns = ['chest_type_id', 'chest_type_name',
                   'chest_region_id', 'chest_region_name',
                   'feature_type_id', 'feature_type_name',
                   'description', 'timestamp', 'user_name', 'machine_name',
                   'coordinate_system', 'lps_to_ijk_transformation_matrix',
                   'spacing', 'origin', 'dimensions'
                   ]

        if len(self.points) > 0:
            # Export points
            columns = ['c1', 'c2', 'c3'] + columns
            df = pd.DataFrame(columns=columns)

            for s in self.points:
                df.loc[s.id] = [s.coordinate[0], s.coordinate[1], s.coordinate[2],
                                s.chest_type, ChestConventions.GetChestTypeName(s.chest_type),
                                s.chest_region, ChestConventions.GetChestRegionName(s.chest_region),
                                s.feature_type, ChestConventions.GetImageFeatureName(s.feature_type),
                                s.description, s.timestamp, s.user_name, s.machine_name,
                                # Common properties
                                self.coordinate_system_str(), self.lps_to_ijk_transformation_matrix_array,
                                self.spacing, self.origin, self.dimensions]

        elif len(self.points) > 0:
            # Export bounding boxes
            columns = ['start1', 'start2', 'start3', 'size1', 'size2', 'size3'] + columns
            df = pd.DataFrame(columns=columns)

            for s in self.bounding_boxes:
                df.loc[s.id] = [s.start[0], s.start[1], s.start[2],
                                s.size[0], s.size[1], s.size[2],
                                s.chest_type, ChestConventions.GetChestTypeName(s.chest_type),
                                s.chest_region, ChestConventions.GetChestRegionName(s.chest_region),
                                s.feature_type, ChestConventions.GetImageFeatureName(s.feature_type),
                                s.description, s.timestamp, s.user_name, s.machine_name,
                                # Common properties
                                self.coordinate_system_str(), self.lps_to_ijk_transformation_matrix_array,
                                self.spacing, self.origin, self.dimensions]

        df.index.name = 'id'
        return df


    @staticmethod
    def __to_xml_vector__(array):
        """ Get the xml representation of a vector of coordinates (<value>elem1</value>, <value>elem2</value>...)
        :param array: vector of values
        :return: xml representation of the vector (<value>elem1</value>, <value>elem2</value>...)
        """
        output = ''
        for i in array:
            output = "{0}<value>{1:g}</value>".format(output, i)
        return output

    @staticmethod
    def __coordinate_system_from_str__(value_str):
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
    def __coordinate_system_to_str__(value_int):
        """ Get the string representation of one of the coordinates systems
        :param value_int: GeometryTopologyData.IJK, GeometryTopologyData.RAS, GeometryTopologyData.LPS...
        :return: string representing the coordinate system ("IJK", "RAS", "LPS"...)
        """
        if value_int == GeometryTopologyData.IJK: return "IJK"
        elif value_int == GeometryTopologyData.RAS: return "RAS"
        elif value_int == GeometryTopologyData.LPS: return "LPS"
        return "UNKNOWN"

    def __read_transformation_matrix__(self, root_xml):
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

    def __write_transformation_matrix__(self, matrix):
        """ Generate an xml text for a 4x4 transformation matrix
        :param matrix: 4x4 list
        :return: xml string (LPStoIJKTransformationMatrix complete node)
        """
        # Flatten the list
        s = ""
        for item in (item for sublist in matrix for item in sublist):
            s += ("<value>{0:g}</value>".format(item))
        return "<LPStoIJKTransformationMatrix>%s</LPStoIJKTransformationMatrix>" % s


class Structure(object):
    def __init__(self, chest_region, chest_type, feature_type, description=None,
                   timestamp=None, user_name=None, machine_name=None):
        """
        :param chest_region: chestRegion Id
        :param chest_type: chestType Id
        :param feature_type: feature type Id (artifacts and others)
        :param description: optional description of the content the element
        :param timestamp: datetime in format "YYYY/MM/dd HH:mm:ss"
        :param user_name: logged username
        :param machine_name: name of the current machine
        """
        self.__id__ = 0
        self.chest_region = chest_region
        self.chest_type = chest_type
        self.feature_type = feature_type
        self.description = description
        self.timestamp = timestamp
        self.user_name = user_name
        self.machine_name = machine_name

    @property
    def id(self):
        return self.__id__

    def get_hash(self):
        """ Get a unique identifier for this structure (string encoding all the fields)
        @return:
        """
        return "%03d_%03d_%03d" % (self.chest_region, self.chest_type, self.feature_type)

    @staticmethod
    def from_xml_node(xml_node):
        """ Return a new instance of a Point object from xml "Point" element
        :param xml_node: xml Point element coming from a "find" instruction
        :return: new instance of the structure
        """
        id = int(xml_node.find("Id").text)
        chest_region = int(xml_node.find("ChestRegion").text)
        chest_type = int(xml_node.find("ChestType").text)
        featureNode = xml_node.find("ImageFeature")
        if featureNode is None:
            feature_type = 0
        else:
            feature_type = int(featureNode.text)

        # Description
        desc = xml_node.find("Description")
        if desc is not None:
            desc = desc.text

        # Timestamp and user info
        timestamp = xml_node.find("Timestamp")
        if timestamp is not None:
            timestamp = timestamp.text
        user_name = xml_node.find("UserName")
        if user_name is not None:
            user_name = user_name.text
        machine_name = xml_node.find("MachineName")
        if machine_name is not None:
            machine_name = machine_name.text

        structure = Structure(chest_region, chest_type, feature_type, description=desc, timestamp=timestamp,
                         user_name=user_name, machine_name=machine_name)
        structure.__id__ = id
        return structure

    def to_xml(self):
        """ Get the xml string representation of the structure that can be appended to a concrete structure (Point,
        BoundingBox, etc)
        :return: xml string representation of the point
        """
        description = ''
        if self.description is not None:
            description = '<Description>%s</Description>' % self.description

        timestamp = ''
        if self.timestamp:
            timestamp = '<Timestamp>%s</Timestamp>' % self.timestamp

        user_name = ''
        if self.user_name:
            user_name = '<UserName>%s</UserName>' % self.user_name

        machine_name = ''
        if self.machine_name:
            machine_name = '<MachineName>%s</MachineName>' % self.machine_name

        return '<Id>%i</Id><ChestRegion>%i</ChestRegion><ChestType>%i</ChestType><ImageFeature>%i</ImageFeature>%s%s%s%s' % \
            (self.__id__, self.chest_region, self.chest_type, self.feature_type, description, timestamp, user_name, machine_name)

    def convert_to_array(self, type_=np.float32):
        """
        Convert the coordinates to a numpy array of the specified type (default: float32)
        Args:
            type_:
        """
        raise NotImplementedError("This method must be implemented by a child class")

    def __str__(self):
        return self.to_xml()



class Point(Structure):
    def __init__(self, chest_region, chest_type, feature_type, coordinate, description=None,
                 timestamp=None, user_name=None, machine_name=None):
        """
        :param id: bounding box id ("autonumeric")
        :param chest_region: chestRegion Id
        :param chest_type: chestType Id
        :param feature_type: feature type Id (artifacts and others)
        :param coordinate: list/tuple of numeric coordinates
        :param description: optional description of the content the element
        :param timestamp: datetime in format "YYYY/MM/dd HH:mm:ss"
        :param user_name: logged username
        :param machine_name: name of the current machine
        :return:
        """
        super(Point, self).__init__(chest_region, chest_type, feature_type, description=description,
                                timestamp=timestamp, user_name=user_name, machine_name=machine_name)

        self.coordinate = coordinate

    def get_hash(self):
        """ Get a unique identifier for this structure (string encoding all the fields)
        @return:
        """
        s = super(Point, self).get_hash()
        for c in self.coordinate:
            s += "_%f" % c
        return s

    @staticmethod
    def from_xml_node(xml_point_node):
        """ Return a new instance of a Point object from xml "Point" element
        :param xml_point_node: xml Point element coming from a "find" instruction
        :return: new instance of Point
        """
        structure = Structure.from_xml_node(xml_point_node)

        coordinates = []
        for coord in xml_point_node.findall("Coordinate/value"):
            coordinates.append(float(coord.text))

        p = Point(structure.chest_region, structure.chest_type, structure.feature_type, coordinates,
                     description=structure.description, timestamp=structure.timestamp, user_name=structure.user_name,
                     machine_name=structure.machine_name)
        p.__id__ = structure.__id__
        return p

    def to_xml(self):
        """ Get the xml string representation of the point
        :return: xml string representation of the point
        """
        # lines = super(FileCatNoEmpty, self).cat(filepath)
        structure = super(Point, self).to_xml()


        coords = GeometryTopologyData.__to_xml_vector__(self.coordinate)
        # description_str = ''
        # if self.description is not None:
        #     description_str = '<Description>%s</Description>' % self.description

        return '<Point>%s<Coordinate>%s</Coordinate></Point>' % (structure, coords)

    def convert_to_array(self, type_=np.float32):
        """
        Convert the coordinates to a numpy array of the specified type (default: float32)
        Args:
            type_:
        """
        self.coordinate = np.array(self.coordinate, dtype=type_)


class BoundingBox(Structure):
    def __init__(self, chest_region, chest_type, feature_type, start, size, description=None,
                 timestamp=None, user_name=None, machine_name=None):
        """
        :param chest_region: chestRegion Id
        :param chest_type: chestType Id
        :param feature_type: feature type Id (artifacts and others)
        :param start: vector of coordinates for the starting point of the Bounding Box
        :param size: vector that contains the size of the bounding box
        :param description: optional description of the content the element
        :param timestamp: datetime in format "YYYY/MM/dd HH:mm:ss"
        :param user_name: logged username
        :param machine_name: name of the current machine
        """
        super(BoundingBox, self).__init__(chest_region, chest_type, feature_type, description=description,
                                timestamp=timestamp, user_name=user_name, machine_name=machine_name)
        self.start = start
        self.size = size

    @property
    def coord2(self):
        """start + size coordinate"""
        return [self.start[0] + self.size[0], self.start[1] + self.size[1], self.start[2] + self.size[2]]

    def get_hash(self):
        """ Get a unique identifier for this structure (string encoding all the fields)
        @return:
        """
        s = super(BoundingBox, self).get_hash()
        for c in self.start:
            s += "_%f" % c
        for c in self.size:
            s += "_%f" % c
        return s

    @staticmethod
    def from_xml_node(xml_bounding_box_node):
        """ Return a new instance of a Point object from xml "BoundingBox" element
        :param xml_bounding_box_node: xml BoundingBox element coming from a "find" instruction
        :return: new instance of BoundingBox
        """
        structure = Structure.from_xml_node(xml_bounding_box_node)
        coordinates_start = []
        for coord in xml_bounding_box_node.findall("Start/value"):
            coordinates_start.append(float(coord.text))
        coordinates_size = []
        for coord in xml_bounding_box_node.findall("Size/value"):
            coordinates_size.append(float(coord.text))

        bb = BoundingBox(structure.chest_region, structure.chest_type, structure.feature_type,
                    coordinates_start, coordinates_size, description=structure.description,
                    timestamp=structure.timestamp, user_name=structure.user_name, machine_name=structure.machine_name)
        bb.__id__ = structure.__id__
        return bb

    def to_xml(self):
        """ Get the xml string representation of the bounding box
        :return: xml string representation of the bounding box
        """
        start_str = GeometryTopologyData.__to_xml_vector__(self.start)
        size_str = GeometryTopologyData.__to_xml_vector__(self.size)
        structure = super(BoundingBox, self).to_xml()

        return '<BoundingBox>%s<Start>%s</Start><Size>%s</Size></BoundingBox>' % (structure, start_str, size_str)

    def convert_to_array(self, type_=np.float32):
        """
        Convert the coordinates to a numpy array of the specified type (default: float32)
        Args:
            type_:
        """
        self.start = np.array(self.start, dtype=type_)
        self.size = np.array(self.size, dtype=type_)