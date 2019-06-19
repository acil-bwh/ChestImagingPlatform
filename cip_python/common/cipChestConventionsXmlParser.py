""" This script is meant to be invoked from Cmake, but it can also be manually invoked from python
to generate the static chest conventions
"""

import os
import os.path as osp
import xml.etree.ElementTree as et
import argparse

xml_conventions_default = os.path.join(os.path.dirname(__file__), "..", "..", "Resources", "ChestConventions.xml")

parser = argparse.ArgumentParser(description='Generate chesst conventions from Xml file.')
parser.add_argument('--xml', dest='xml_conventions_file', metavar='<string>', default=xml_conventions_default,
                    help='xml conventions file')
parser.add_argument('--in_cxx', dest='template_cxx_path', metavar='<string>',default=None,
                    help='Input template .h file')
parser.add_argument('--out_cxx', dest='output_path_cxx', metavar='<string>',default=None,
                    help='output file path for .h')
parser.add_argument('--in_python', dest='template_python_path', metavar='<string>',default=None,
                    help='Input template python file')
parser.add_argument('--out_python_source', dest='out_python_source', metavar='<string>',default=None,
                    help='output file path for python (source folder)')
parser.add_argument('--out_python_bin', dest='out_python_bin', metavar='<string>',default=None,
                    help='output file path for python (destination bin folder)')

op =  parser.parse_args()

if op.template_cxx_path is not None and osp.isfile(op.template_cxx_path):
    with open(op.template_cxx_path, "r") as f:
        template_cxx = f.read()
else:
    template_cxx = ""

if op.template_python_path is not None and osp.isfile(op.template_python_path):
    with open(op.template_python_path, "r") as f:
        template_python = f.read()
else:
    template_python = ""

# Read the XML
with open(op.xml_conventions_file, "r") as f:
    xml_root = et.fromstring(f.read())

replacements = (
    ("ChestTypes/ChestType", "//##CHEST_TYPE_ENUM##"),
    ("ChestRegions/ChestRegion", "//##CHEST_REGION_ENUM##"),
    ("ImageFeatures/ImageFeature", "//##IMAGE_FEATURE_ENUM##"),
    ("ReturnCodes/ReturnCode", "//##RETURN_CODE_ENUM##"),
    ("Planes/Plane", "//##PLANE_ENUM##")
)
for replacement in replacements:
    replacement_text_cxx = ""
    replacement_text_python = ""
    nodes_query = replacement[0]
    replacement_tag = replacement[1]
    for node in xml_root.findall(nodes_query):
        replacement_text_cxx += "    {},\n".format(node.find("Code").text)
        replacement_text_python += "    {} = {}\n".format(node.find("Code").text, node.find("Id").text)

    template_cxx = template_cxx.replace(replacement_tag, replacement_text_cxx)
    template_python = template_python.replace(replacement_tag, replacement_text_python)

# Replace collections (unsigned char codes)
replacement_text_cxx = ""
replacements = (
    "ChestTypes/ChestType",
    "ChestRegions/ChestRegion",
    "ImageFeatures/ImageFeature",
    "Planes/Plane"
)
for replacement in replacements:
    for node in xml_root.findall(replacement):
        replacement_text_cxx += "            {}.push_back( (unsigned char)( {} ) );\n".format(replacement.split("/")[0],
                                                                                              node.find("Code").text)
    replacement_text_cxx += "\n"

# Replace collections (string names)
replacement_text_cxx += "\n"
for replacement in replacements:
    for node in xml_root.findall(replacement):
        replacement_text_cxx += "            {}Names.push_back( \"{}\" );\n".format(replacement.split("/")[1],
                                                                                    node.find("Name").text)
    replacement_text_cxx += "\n"

# PhenotypeNames
replacement_text_cxx += "\n"
replacements = (
    "BodyCompositionPhenotypeNames",
    "ParenchymaPhenotypeNames",
    "PulmonaryVasculaturePhenotypeNames",
)
for replacement in replacements:
    for node in xml_root.findall("{}/Name".format(replacement)):
        replacement_text_cxx += "            {}.push_back( \"{}\" );\n".format(replacement, node.text)
    replacement_text_cxx += "\n"

# Hierarchy map
replacement_text_cxx += "\n"
replacement_text_cxx += \
"""         // For the hierarchical relationships, leftness and rightness
            // are respected before any relationship that transcends
            // leftness or rightness. For example left lower third maps to
            // left lung, not lower third, etc. The exception to this rule
            // is that both left and right lungs are subordinate to
            // WHOLELUNG, not LEFT and RIGHT\n"""
i = 0
for hierarchy_node in xml_root.findall("ChestRegionHierarchyMap/Hierarchy"):
    node_text = "            std::vector<unsigned char> tmp_{};\n".format(i)
    for parent in hierarchy_node.findall("Parents/Parent"):
        node_text += "            tmp_{}.push_back((unsigned char){});\n".format(i, parent.text)
    node_text += "            ChestRegionHierarchyMap.insert(Region_Pair((unsigned char)({}), tmp_{}));\n".format(hierarchy_node.find("Child").text, i)
    replacement_text_cxx += node_text
    i += 1

# Colors
i = 0
replacement_text_cxx += "\n"
replacements = (
    "ChestTypes/ChestType",
    "ChestRegions/ChestRegion")

for replacement in replacements:
    for node in xml_root.findall(replacement):
        color = node.find("Color").text.split(";")
        # replacement_text += "            double* t092 = new double[3]; t092[0] = 0.03; t092[1] = 0.03; t092[2] = 0.04; ChestTypeColors.push_back( t092 ); " \
        replacement_text_cxx += "            double* t{0:03} = new double[3]; t{0:03}[0] = {1}; t{0:03}[1] = {2}; t{0:03}[2] = {3}; {4}Colors.push_back( t{0:03} );\n". \
            format(i, color[0], color[1], color[2], replacement.split("/")[1])
        i += 1

    replacement_text_cxx += "\n"

template_cxx = template_cxx.replace("//##STRUCTURES##", replacement_text_cxx)

# Write output result files
if op.output_path_cxx:
    with open(op.output_path_cxx, "w") as f:
        f.write(template_cxx)
if op.out_python_source:
    with open(op.out_python_source, "w") as f:
        f.write(template_python)
if op.out_python_bin:
    with open(op.out_python_bin, "w") as f:
        f.write(template_python)

print("Convention Files generated succesfully.")
if op.output_path_cxx:
    print("C++ Output path: %s" % op.output_path_cxx)
if op.out_python_source:
    print("Python source path: %s" % op.out_python_source)
if op.out_python_bin:
    print("Python dest binary folder path: %s" % op.out_python_bin)


def generate_colortable_file():
    import cip_python.common as common
    output = ""
    for rkey, rvalue in common.ChestConventions.ChestRegionsCollection.items():
        for tkey, tvalue in common.ChestConventions.ChestTypesCollection.items():
            code = common.ChestConventions.GetValueFromChestRegionAndType(rkey, tkey)
            description = "{}-{}".format(rvalue[1], tvalue[1])
            color = common.ChestConventions.GetColorFromChestRegionChestType(rkey, tkey)
            output += "{} {}-{} {} {} {} 255\n".format(code, rvalue[1], tvalue[1], int(color[0] * 255), int(color[1] * 255),
                                          int(color[2] * 255))