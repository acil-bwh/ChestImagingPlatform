import os.path as path
from collections import OrderedDict
import xml.etree.ElementTree as et
from future.utils import iteritems, itervalues

from cip_python.common.chest_conventions_static import *

class ChestConventionsInitializer(object):
    #root_xml_path = "/Users/jonieva/Projects/CIP/Resources/ChestConventions/"
    root_xml_path = path.realpath(path.join(path.dirname(__file__), "..", "..", "Resources", "ChestConventions.xml"))
    __xml_conventions__ = None

    __chest_regions__ = None
    __chest_types__ = None
    __image_features__ = None
    __planes__ = None
    __chest_regions_hierarchy__ = None
    __preconfigured_colors__ = None
    __body_composition_phenotype_names__ = None
    __parenchyma_phenotype_names__ = None
    __pulmonary_vasculature_phenotype_names__ = None
    __airway_phenotype_names__ = None
    __biomechanical_phenotype_names__ = None
    __fissure_phenotype_names__ = None

    @staticmethod
    def xml_root_conventions():
        if ChestConventionsInitializer.__xml_conventions__ is None:
            with open(ChestConventionsInitializer.root_xml_path, 'rb') as f:
                xml = f.read()
                ChestConventionsInitializer.__xml_conventions__ = et.fromstring(xml)
        return ChestConventionsInitializer.__xml_conventions__

    @staticmethod
    def __loadCSV__(file_name):
        """ Return a "list of lists of strings" with all the rows read from a csv file
        Parameters
        ----------
        file_name: name of the file (including the extension). The full path will be concatenated with Root_Folder

        Returns
        -------
        Lists of lists (every row of the csv file will be a list of strings
        """
        import csv
        folder = path.dirname(ChestConventionsInitializer.root_xml_path)
        csv_file_path = path.join(folder, file_name)
        if not path.exists(csv_file_path):
            raise NameError("Resources file not found ({})".format(csv_file_path))
        with open(csv_file_path, 'rb') as f:
            reader = csv.reader(f)
            # Return the concatenation of all the rows in a list
            return [row for row in reader]

    @staticmethod
    def chest_regions():
        if ChestConventionsInitializer.__chest_regions__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__chest_regions__ = OrderedDict()
            parent = root.find("ChestRegions")
            chest_regions_enum = ChestRegion.elems_as_dictionary()
            for xml_region in parent.findall("ChestRegion"):
                elem_id = int(xml_region.find("Id").text)
                if elem_id not in chest_regions_enum:
                    raise AttributeError("The key {0} in ChestRegions does not belong to the enumeration"
                                         .format(elem_id))
                ChestConventionsInitializer.__chest_regions__[elem_id] = (
                    xml_region.find("Code").text,
                    xml_region.find("Name").text,
                    list(map(lambda s: float(s), xml_region.find("Color").text.split(";")))
                )

        return ChestConventionsInitializer.__chest_regions__

    @staticmethod
    def chest_regions_hierarchy():
        if ChestConventionsInitializer.__chest_regions_hierarchy__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__chest_regions_hierarchy__ = {}
            parent = root.find("ChestRegionHierarchyMap")
            for hierarchy_node in parent.findall("Hierarchy"):
                c = eval("ChestRegion.{}".format(hierarchy_node.find("Child").text))
                # Collection of parents (a child may have more than one parent)
                parents = []
                for parent in hierarchy_node.findall("Parents/Parent"):
                    parents.append(eval("ChestRegion.{}".format(parent.text)))
                ChestConventionsInitializer.__chest_regions_hierarchy__[c] = parents
        return ChestConventionsInitializer.__chest_regions_hierarchy__


    @staticmethod
    def chest_types():
        if ChestConventionsInitializer.__chest_types__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__chest_types__ = OrderedDict()
            parent = root.find("ChestTypes")
            chest_types_enum = ChestType.elems_as_dictionary()
            for xml_type in parent.findall("ChestType"):
                elem_id = int(xml_type.find("Id").text)
                if elem_id not in chest_types_enum:
                    raise AttributeError("The key {0} in ChestTypes does not belong to the enumeration"
                                         .format(elem_id))
                try:
                    ChestConventionsInitializer.__chest_types__[elem_id] = (
                        xml_type.find("Code").text,
                        xml_type.find("Name").text,
                        list(map(lambda s: float(s), xml_type.find("Color").text.split(";")))
                    )
                except Exception as ex:
                    print ("Error in {}".format(elem_id))
                    raise ex

        return ChestConventionsInitializer.__chest_types__

    @staticmethod
    def image_features():
        if ChestConventionsInitializer.__image_features__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__image_features__ = OrderedDict()
            parent = root.find("ImageFeatures")
            image_features_enum = ImageFeature.elems_as_dictionary()
            for xml_type in parent.findall("ImageFeature"):
                elem_id = int(xml_type.find("Id").text)
                if elem_id not in image_features_enum:
                    raise AttributeError("The key {0} in ImageFeatures does not belong to the enumeration"
                                         .format(elem_id))
                ChestConventionsInitializer.__image_features__[elem_id] = (
                    xml_type.find("Code").text,
                    xml_type.find("Name").text
                )
        return ChestConventionsInitializer.__image_features__

    @staticmethod
    def planes():
        if ChestConventionsInitializer.__planes__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__planes__ = OrderedDict()
            parent = root.find("Planes")
            enum = Plane.elems_as_dictionary()
            for xml_type in parent.findall("Plane"):
                elem_id = int(xml_type.find("Id").text)
                if elem_id not in enum:
                    raise AttributeError("The key {0} in Planes does not belong to the enumeration"
                                         .format(elem_id))
                ChestConventionsInitializer.__planes__[elem_id] = (
                    xml_type.find("Code").text,
                    xml_type.find("Name").text
                )
        return ChestConventionsInitializer.__planes__

    @staticmethod
    def preconfigured_colors():
        if ChestConventionsInitializer.__preconfigured_colors__ is None:
            ChestConventionsInitializer.__preconfigured_colors__ = OrderedDict()
            rows = ChestConventionsInitializer.__loadCSV__("PreconfiguredColors.csv")
            for row in rows:
                region = eval('ChestRegion.' + row[0].strip())
                _type = eval('ChestType.' + row[1].strip())
                ChestConventionsInitializer.__preconfigured_colors__[(region, _type)] = \
                    (float(row[2].strip()), float(row[3].strip()), float(row[4].strip()))
        return ChestConventionsInitializer.__preconfigured_colors__

    @staticmethod
    def body_composition_phenotype_names():
        if ChestConventionsInitializer.__body_composition_phenotype_names__ is None:
                root = ChestConventionsInitializer.xml_root_conventions()
                ChestConventionsInitializer.__body_composition_phenotype_names__ = list()
                parent = root.find("BodyCompositionPhenotypeNames")
                list(map(lambda n: ChestConventionsInitializer.__body_composition_phenotype_names__.append(n.text),
                    parent.findall("Name")))
        return ChestConventionsInitializer.__body_composition_phenotype_names__

    @staticmethod
    def parenchyma_phenotype_names():
        if ChestConventionsInitializer.__parenchyma_phenotype_names__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__parenchyma_phenotype_names__ = list()
            parent = root.find("ParenchymaPhenotypeNames")
            list(map(lambda n: ChestConventionsInitializer.__parenchyma_phenotype_names__.append(n.text),
                parent.findall("Name")))
        return ChestConventionsInitializer.__parenchyma_phenotype_names__

    @staticmethod
    def pulmonary_vasculature_phenotype_names():
        if ChestConventionsInitializer.__pulmonary_vasculature_phenotype_names__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__pulmonary_vasculature_phenotype_names__ = list()
            parent = root.find("PulmonaryVasculaturePhenotypeNames")
            list(map(lambda n: ChestConventionsInitializer.__pulmonary_vasculature_phenotype_names__.append(n.text),
                parent.findall("Name")))
        return ChestConventionsInitializer.__pulmonary_vasculature_phenotype_names__

    @staticmethod
    def airway_phenotype_names():
        if ChestConventionsInitializer.__airway_phenotype_names__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__airway_phenotype_names__ = list()
            parent = root.find("AirwayPhenotypeNames")
            list(map(lambda n: ChestConventionsInitializer.__airway_phenotype_names__.append(n.text),
                parent.findall("Name")))
        return ChestConventionsInitializer.__airway_phenotype_names__

    @staticmethod
    def biomechanical_phenotype_names():
        if ChestConventionsInitializer.__biomechanical_phenotype_names__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__biomechanical_phenotype_names__ = list()
            parent = root.find("BiomechanicalPhenotypeNames")
            list(map(lambda n: ChestConventionsInitializer.__biomechanical_phenotype_names__.append(n.text),
                parent.findall("Name")))
        return ChestConventionsInitializer.__biomechanical_phenotype_names__

    @staticmethod    
    def fissure_phenotype_names():
        if ChestConventionsInitializer.__fissure_phenotype_names__ is None:
            root = ChestConventionsInitializer.xml_root_conventions()
            ChestConventionsInitializer.__fissure_phenotype_names__ = list()
            parent = root.find("FissurePhenotypeNames")
            list(map(lambda n: ChestConventionsInitializer.__fissure_phenotype_names__.append(n.text),
                parent.findall("Name")))
        return ChestConventionsInitializer.__fissure_phenotype_names__

    
#############################
# CHEST CONVENTIONS
#############################
class ChestConventions(object):
    ChestRegionsCollection = ChestConventionsInitializer.chest_regions()      # 1: "WHOLELUNG", "WholeLung", [0.42, 0.38, 0.75]
    ChestRegionsHierarchyCollection = ChestConventionsInitializer.chest_regions_hierarchy()     # LEFTSUPERIORLOBE, LEFTLUNG
    ChestTypesCollection = ChestConventionsInitializer.chest_types()          # 1:, "NORMALPARENCHYMA", "NormalParenchyma", [0.99, 0.99, 0.99]
    ImageFeaturesCollection = ChestConventionsInitializer.image_features()  # 1: "CTARTIFACT", "CTArtifact"
    PlanesCollection = ChestConventionsInitializer.planes()  # 1: "AXIAL", "Axial"
    PreconfiguredColors = ChestConventionsInitializer.preconfigured_colors()
    #
    BodyCompositionPhenotypeNames = ChestConventionsInitializer.body_composition_phenotype_names()   # List of strings
    ParenchymaPhenotypeNames = ChestConventionsInitializer.parenchyma_phenotype_names()   # List of strings
    PulmonaryVasculaturePhenotypeNames = ChestConventionsInitializer.pulmonary_vasculature_phenotype_names()   # List of strings
    AirwayPhenotypeNames = ChestConventionsInitializer.airway_phenotype_names()   # List of strings
    BiomechanicalPhenotypeNames = ChestConventionsInitializer.biomechanical_phenotype_names() #List of strings
    FissurePhenotypeNames = ChestConventionsInitializer.fissure_phenotype_names() #List of strings    

    @staticmethod
    def GetNumberOfEnumeratedChestRegions():
        return len(ChestConventions.ChestRegionsCollection)

    @staticmethod
    def GetNumberOfEnumeratedChestTypes():
        return len(ChestConventions.ChestTypesCollection)

    @staticmethod
    def GetNumberOfEnumeratedImageFeatures():
        return len(ChestConventions.ImageFeaturesCollection)

    @staticmethod
    def GetNumberOfEnumeratedPlanes():
        return len(ChestConventions.PlanesCollection)

    @staticmethod
    def CheckSubordinateSuperiorChestRegionRelationship(subordinate, superior):
        """
        This method checks if the chest region 'superior' is a predecessor of 'subordinate' in the whole hierarchy
        :param subordinate: ChestRegion code (int)
        :param superior: list of ChestRegion code (int)
        :return: boolean
        """
        # Base cases
        if subordinate == superior:
            return True
        if ChestRegion.UNDEFINEDREGION in (subordinate, superior):
            return False
        if subordinate not in ChestConventions.ChestRegionsHierarchyCollection:
            # Bastard node (no parents known, so it's not part of the hierarchy)
            return False

        # Loop over the whole hierarchy
        # Get parents of the child node (make a copy of the list, as ChestRegionsHierarchyCollection is static)
        parents = list(ChestConventions.ChestRegionsHierarchyCollection[subordinate])
        while len(parents) > 0:
            # Get the last element of the list (Deep breath search)
            parent = parents.pop()
            if parent == superior:
                # We got to a parent of the original child
                return True
            if parent in ChestConventions.ChestRegionsHierarchyCollection:
                # Add to the list to inspect the whole set of parents
                parents.extend(ChestConventions.ChestRegionsHierarchyCollection[parent])
        # Parent not found
        return False


    @staticmethod
    def GetChestRegionFromValue(value):
        """
        Given an unsigned short value, this method will compute the 8-bit region value corresponding to the input
        :param value: int
        :return: int
        """
        return 255 & value  # Less significant byte

    @staticmethod
    def GetChestTypeFromColor(color):
        """
        The 'color' param is assumed to have three components, each in
        the interval [0,1]. All chest type colors will be tested until a
        color match is found. If no match is found, 'UNDEFINEDTYPYE'
        will be returned
        :param color: tuple/array of 3 components
        :return: int
        """
        for key, value in iteritems(ChestConventions.ChestTypesCollection):
            if value[1] == color[0] and value[2] == color[1] and value[3] == color[2]:
                return key
        # Not found
        return ChestType.UNDEFINEDTYPE

    @staticmethod
    def GetChestRegionFromColor(color):
        """
        The 'color' param is assumed to have three components, each in
        the interval [0,1]. All chest type colors will be tested until a
        color match is found. If no match is found, 'UNDEFINEDREGION'
        will be returned
        :param color: tuple/array of 3 components
        :return: int
        """
        for key, value in iteritems(ChestConventions.ChestRegionsCollection):
            if value[1] == color[0] and value[2] == color[1] and value[3] == color[2]:
                return key
        # Not found
        return ChestRegion.UNDEFINEDREGION

    @staticmethod
    def GetChestTypeFromValue(value):
        """
        Given an unsigned short value, this method will compute the 8-bit type value corresponding to the input
        :param value: int
        :return: int
        """
        return value >> 8   # Most significant byte


    @staticmethod
    def GetChestWildCardName():
        return "WildCard"

    @staticmethod
    def GetChestTypeName(whichType):
        """
         Given an int value corresponding to a chest type, this method will return the string name equivalent.
        :param whichType: int
        :return: string code (descriptive, ex: WholeLung")
        """
        if whichType not in ChestConventions.ChestTypesCollection:
            #raise IndexError("Key {0} is not a valid ChestType".format(whichType))
            # C++ compatibility:
            return ChestConventions.GetChestTypeName(ChestType.UNDEFINEDTYPE)
        return ChestConventions.ChestTypesCollection[whichType][1]

    @staticmethod
    def GetChestTypeColor(whichType, color=None):
        """
        Get the color for a ChestType.
        If color has some value, it will suppose to be a list where the color will be stored (just for compatibility purposes).
        In any case, the color will be returned as the result of the function
        :param whichType: int or string code
        :param color: 3-tuple where the result will be stored (for C++ compatibility reasons)
        :return: 3-tuple result
        """
        if type(whichType) == str:
            whichType = ChestConventions.GetChestTypeValueFromName(whichType)

        if whichType not in ChestConventions.ChestTypesCollection:
            raise IndexError("Key {0} is not a valid ChestType".format(whichType))
        col = ChestConventions.ChestTypesCollection[whichType][2]
        if color is not None:
            color[0] = col[0]
            color[1] = col[1]
            color[2] = col[2]
        return col


    @staticmethod
    def GetChestRegionColor(whichRegion, color=None):
        """
        Get the color for a ChestRegion.
        If color has some value, it will suppose to be a list where the color will be stored (just for compatibility purposes).
        In any case, the color will be returned as the result of the function
        :param whichRegion: int
        :param color: 3-tuple color
        :return:
        """
        """
        Parameters
        ----------
        whichRegion
        color

        Returns
        -------
        3-Tuple with the color
        """
        if whichRegion not in ChestConventions.ChestRegionsCollection:
            raise IndexError("Key {0} is not a valid ChestRegion".format(whichRegion))
        col = ChestConventions.ChestRegionsCollection[whichRegion][2]
        if color is not None:
            color[0] = col[0]
            color[1] = col[1]
            color[2] = col[2]
        return col

    @staticmethod
    def GetColorFromChestRegionChestType(whichRegion, whichType, color=None):
        """ Get the color for a particular Region-Type.
        It can be preconfigured, a single region/type or the mean value.
        If color has some value, it will suppose to be a list where the color will be stored (just for compatibility purposes).
        In any case, the color will be returned as the result of the function
        Parameters
        ----------
        whichRegion
        whichType
        color

        Returns
        -------
        3-Tuple with the color
        """
        # Check first if the combination is preconfigured
        if (whichRegion, whichType) not in ChestConventions.PreconfiguredColors:
            col = ChestConventions.PreconfiguredColors[(whichRegion, whichType)]
        elif whichRegion == ChestRegion.UNDEFINEDREGION:
            col = ChestConventions.GetChestTypeColor(whichType)
        elif whichType == ChestType.UNDEFINEDTYPE:
            col = ChestConventions.GetChestRegionColor(whichRegion)
        else:
            # Just take the average of two colors
            reg_color = ChestConventions.GetChestRegionColor(whichRegion)
            type_color = ChestConventions.GetChestTypeColor(whichType)
            print (reg_color)
            print (type_color)
            col = ((reg_color[0] + type_color[0]) / 2.0,
                   (reg_color[1] + type_color[1]) / 2.0,
                   (reg_color[2] + type_color[2]) / 2.0)

        if color is not None:
            color[0] = col[0]
            color[1] = col[1]
            color[2] = col[2]
        return col


    @staticmethod
    def GetChestRegionName(whichRegion):
        """
        Given an unsigned char value corresponding to a chest region, this method will return the string name equivalent
        :param whichRegion: int
        :return:
        """
        if whichRegion not in ChestConventions.ChestRegionsCollection:
            #raise IndexError("Key {0} is not a valid ChestRegion".format(whichRegion))
            # C++ compatibility:
            return ChestConventions.GetChestRegionName(ChestRegion.UNDEFINEDREGION)
        return ChestConventions.ChestRegionsCollection[whichRegion][1]

    @staticmethod
    def GetChestRegionNameFromValue(value):
        """
        Given an int code, this method will return the string name of the corresponding chest region.
        C++ compatibility
        :param value: int
        :return: descriptive string (Ex: "WholeLung")
        """
        return ChestConventions.GetChestRegionName(value)


    @staticmethod
    def GetChestTypeNameFromValue(value):
        """
        Given an int code, this method will return the string name of the corresponding chest type.
        C++ compatibility
        :param value: int
        :return: descriptive string (Ex: "WholeLung")
        """
        return ChestConventions.GetChestTypeName(value)

    @staticmethod
    def GetValueFromChestRegionAndType(region, type):
        """
        Get a single coded int from an int region and an int type, with the combination of most-less significant byte
        :param region: int
        :param type: int
        :return: int
        """
        return (type << 8) + region

    @staticmethod
    def GetChestRegionValueFromName(regionString):
        """
        Given a string identifying one of the enumerated chest regions, this method will return the int code equivalent.
         If no match is found, the method will return UNDEFINEDREGION
        :param regionString: string (case-insensitve, but compare with string descriptive names (ex: WholeLung))
        :return: int
        """
        for key,value in iteritems(ChestConventions.ChestRegionsCollection):
            if value[1].lower() == regionString.lower():
                return key
        raise KeyError("Region not found: " + regionString)

    @staticmethod
    def GetChestTypeValueFromName(typeString):
        """
        Given a string identifying one of the enumerated chest types, this method will return the int code equivalent.
         If no match is found, the method will return UNDEFINEDTYPE
        :param regionString: string (case-insensitve, but compare with string descriptive names (ex: WholeLung))
        :return: int
        """
        for key, value in iteritems(ChestConventions.ChestTypesCollection):
            if value[1].lower() == typeString.lower():
                return key
        raise KeyError("Type not found: " + typeString)

    @staticmethod
    def GetPlaneValueFromName(planeString):
        """
        Given a string identifying one of the enumerated planes, this method will return the int code equivalent.
         If no match is found, the method will return UNDEFINEDPLANE
        :param planeString: string (case-insensitve, but compare with string descriptive names (ex: WholeLung))
        :return: int
        """
        for key, value in iteritems(ChestConventions.PlanesCollection):
            if value[1].lower() == planeString.lower():
                return key
        raise KeyError("Plane not found: {}".format(planeString))

    @staticmethod
    def GetChestRegion(i):
        """DEPRECATED. Just for C++ compatibility"""
        return i

    @staticmethod
    def GetChestType(i):
        """DEPRECATED. Just for C++ compatibility"""
        return i

    @staticmethod
    def GetImageFeature(i):
        """DEPRECATED. Just for C++ compatibility"""
        return i

    @staticmethod
    def GetImageFeatureName(whichFeature):
        """
         Given an int value corresponding to an ImageFeature, this method will return the string name equivalent
        :param whichFeature: int
        :return: descriptive string
        """
        if whichFeature not in ChestConventions.ImageFeaturesCollection:
            #raise IndexError("Key {0} is not a valid Image Feature".format(whichFeature))
            # C++ compatibility:
            return ChestConventions.GetImageFeatureName(ImageFeature.UNDEFINEDFEATURE)
        return ChestConventions.ImageFeaturesCollection[whichFeature][1]

    @staticmethod
    def GetPlaneName(whichPlane):
        """
         Given an int value corresponding to a Plane, this method will return the string name equivalent
        :param whichPlane: int
        :return: descriptive string
        """
        if whichPlane not in ChestConventions.PlanesCollection:
            raise IndexError("Key '{0}' is not a valid Plane".format(whichPlane))
            # C++ compatibility:
            # return ChestConventions.GetImageFeatureName(ImageFeature.UNDEFINEDFEATURE)
        return ChestConventions.PlanesCollection[whichPlane][1]

    @staticmethod
    def IsBodyCompositionPhenotypeName(phenotypeName):
        """
        Returns true if the passed phenotype is among the allowed body composition phenotype names.
        :param phenotypeName: str
        :return: boolean
        """
        return phenotypeName in ChestConventions.BodyCompositionPhenotypeNames

    @staticmethod
    def IsParenchymaPhenotypeName(phenotypeName):
        """
        Returns true if the passed phenotype is among the allowed parenchyma phenotype names.
        :param phenotypeName: str
        :return: boolean
        """
        return phenotypeName in ChestConventions.ParenchymaPhenotypeNames

    @staticmethod
    def IsBiomechanicalPhenotypeName(phenotypeName):
        """
        Returns true if the passed phenotype is among the allowed biomechanical phenotype names.
        :param phenotypeName: str
        :return: boolean
        """
        return phenotypeName in ChestConventions.BiomechanicalPhenotypeNames

    @staticmethod
    def IsFissurePhenotypeName(phenotypeName):
        """
        Returns true if the passed phenotype is among the allowed fissure phenotype names.
        :param phenotypeName: str
        :return: boolean
        """
        return phenotypeName in ChestConventions.FissurePhenotypeNames

    @staticmethod
    def IsPulmonaryVasculaturePhenotypeName(phenotypeName):
        """
        Returns true if the passed phenotype is among the allowed body composition phenotype names.
        :param phenotypeName: str
        :return: boolean
        """
        return phenotypeName in ChestConventions.PulmonaryVasculaturePhenotypeNames

    @staticmethod
    def IsAirwayPhenotypeName(phenotypeName):
        """
            Returns true if the passed phenotype is among the allowed body composition phenotype names.
            :param phenotypeName: str
            :return: boolean
            """
        return phenotypeName in ChestConventions.AirwayPhenotypeNames


    @staticmethod
    def IsHistogramPhenotypeName(phenotypeName):
        """
        DEPRECATED. Not used so far
        :param phenotypeName:
        :return: False
        """
        return False

    # NOTE: In case there are more phenotypes lists, they should be added to the GetPhenotypeNamesLists() function

    @staticmethod
    def IsPhenotypeName(phenotypeName):
        """
        True if phenotypeName is in any of the current phenotype lists
        :param phenotypeName: str
        :return: Boolean
        """
        for l in ChestConventions.GetPhenotypeNamesLists():
            if phenotypeName in l: return True
        return False

    @staticmethod
    def GetPhenotypeNamesLists():
        """
        Get all the current phenotype lists
        :return: list
        """
        return [ChestConventions.BodyCompositionPhenotypeNames, ChestConventions.ParenchymaPhenotypeNames,
                ChestConventions.PulmonaryVasculaturePhenotypeNames, ChestConventions.AirwayPhenotypeNames,
		ChestConventions.BiomechanicalPhenotypeNames, ChestConventions.FissurePhenotypeNames]

    @staticmethod
    def IsChestType(chestType):
        """
        True if the code matches to a current chest type.
        Please note that the code can be either a descriptive string or an int code for C++ compatibility
        :param chestType: string (for C++ compatibility) or int code
        :return: Boolean
        """
        if type(chestType) == str:
            # Loop over the exact Chest Types description names (C++ compatibility)
            for t in itervalues(ChestConventions.ChestTypesCollection):
                if chestType == t[1]:
                    return True
            return False

        else:
            # Int code
            return chestType in ChestConventions.ChestTypesCollection

    @staticmethod
    def IsChestRegion(chestRegion):
        """
        True if the code matches to a current chest retgion.
        Please note that the code can be either a descriptive string or an int code for C++ compatibility
        :param chestRegion: string (for C++ compatibility) or int code
        :return: Boolean
        """
        if type(chestRegion) == str:
            # Loop over the exact Chest Region description names (C++ compatibility)
            for r in itervalues(ChestConventions.ChestRegionsCollection):
                if chestRegion == r[1]:
                    return True
            return False
        else:
            # Int code
            return chestRegion in ChestConventions.ChestRegionsCollection


      


#############################
# SANITY CHECKS
#############################
# def test_chest_conventions():
#    import CppHeaderParser     # Import here in order not to force the CppHeaderParser module to use ChestConventions (it's just needed for testing and it's not a standard module)

# p = "/Users/jonieva/Projects/CIP/Common/cipChestConventions.h"
# cppHeader = CppHeaderParser.CppHeader(p)
# c_chest_conventions = cppHeader.classes["ChestConventions"]

# def compare_c_python_enum(enum_name, c_enum, p_enum):
#     """ Make sure that all the values in a C++ enumeration are the same in Python
#     Parameters
#     ----------
#     enum_name: name of the enumeration
#     c_enum: C++ enumeration
#     p_enum: Python enumeration
#     """
#     for elem in c_enum:
#         name = elem['name']
#         int_value = elem['value']
#         if int_value not in p_enum:
#             raise Exception("Error in {0}: Key {1} was found in C++ object but not in Python".format(enum_name, int_value))
#         if p_enum[int_value] != name:
#             raise Exception("Error in {0}: {0}[{1}] (C++) = {2}, but {0}[{1}] (Python) = {3}".format(
#                 enum_name, int_value, name, p_enum[int_value]))
#
# def compare_python_c_enum(enum_name, p_enum, c_enum):
#     """ Make sure that all the values in a Python enumeration are the same in C++
#     Parameters
#     ----------
#     enum_name: name of the enumeration
#     p_enum: Python enumeration
#     c_enum: C++ enumeration
#     """
#     for int_value, description in p_enum.iteritems():
#         found = False
#         for item in c_enum:
#             if item['value'] == int_value:
#                 found = True
#                 if item['name'] != description:
#                     raise Exception("Error in {0}. {0}[{1}} (Python) = {2}, but {0}[{1}] (C++) = {3}".format(
#                         enum_name, int_value, description, item['name']))
#                 break
#         if not found:
#             raise Exception("Error in {0}. Elem '{1}' does not exist in C++".format(enum_name, description))
#
# def compare_python_c_methods(p_methods, c_methods):
#     """ Make sure all the python methods in ChestConventions are the same in Python that in C++
#     Parameters
#     ----------
#     p_methods: Python methods
#     c_methods: C++ methods
#     """
#     for p_method in p_methods:
#         found = False
#         p_name = p_method.func_name
#         for c_method in c_methods:
#             c_name = c_method["name"]
#             if c_name == p_name:
#                 # Matching method found in C++. Check the parameters
#                 found = True
#                 p_args = p_method.func_code.co_varnames
#                 c_args = c_method["parameters"]
#                 if len(p_args) != len(c_args):
#                     raise Exception ("Method '{0}' has {1} parameters in Python and {2} in C++".format(p_name,
#                                                                                             len(p_args), len(c_args)))
#                 for i in range(len(p_args)):
#                     if p_args[i] != c_args[i]["name"]:
#                         raise Exception("The parameter number {0} in Python method '{1}' is '{2}', while in C++ it's '{3}'".
#                                         format(i, p_name, p_args[i], c_args[i]["name"]))
#                 break
#         if not found:
#             raise Exception("Python method '{0}' was not found in C++".format(p_name))
#
# def compare_c_python_methods(c_methods, p_methods):
#     """ Make sure all the python methods in ChestConventions are the same in Python that in C++
#     Parameters
#     ----------
#     c_methods: C++ methods
#     p_methods: Python methods
#     """
#     for c_method in c_methods:
#         if c_method["destructor"] or c_method["constructor"]:
#             continue
#         found = False
#         c_name = c_method["name"]
#         for p_method in p_methods:
#             p_name = p_method.func_name
#             if c_name == p_name:
#                 # Matching method found in Python. Check the parameters
#                 found = True
#                 c_args = c_method["parameters"]
#                 p_args = p_method.func_code.co_varnames
#                 if len(p_args) != len(c_args):
#                     raise Exception ("Method '{0}' has {1} parameters in Python and {2} in C++".format(p_name,
#                                                                                             len(p_args), len(c_args)))
#                 for i in range(len(p_args)):
#                     if p_args[i] != c_args[i]["name"]:
#                         raise Exception("The parameter number {0} in Python method '{1}' is '{2}', while in C++ it's '{3}'".
#                                         format(i, p_name, p_args[i], c_args[i]["name"]))
#                 break
#         if not found:
#             raise Exception("C++ method '{0}' was not found in Python".format(c_name))

# def total_checking():
#     # Go through all the enumerations in C++ and make sure we have the same values in Python
#     for i in range(len(cppHeader.enums)):
#         c_enum = cppHeader.enums[i]["values"]
#         enum_name = cppHeader.enums[i]['name']
#         p_enum = eval("{0}.elems_as_dictionary()".format(enum_name))
#         compare_c_python_enum(enum_name, c_enum, p_enum)
#         compare_python_c_enum(enum_name, p_enum, c_enum)
#         print(cppHeader.enums[i]['name'] + "...OK")
#
#     # Make sure that all the methods in ChestConventions in C++ are implemented in Python and viceversa
#     p_methods = [f[1] for f in inspect.getmembers(ChestConventions, inspect.isfunction)]
#     c_methods = c_chest_conventions.get_all_methods()
#
#     compare_c_python_methods(c_methods, p_methods)
#     print("C++ --> Python checking...OK")
#
#     compare_python_c_methods(p_methods, c_methods)
#     print("Python --> C++ checking...OK")
#

