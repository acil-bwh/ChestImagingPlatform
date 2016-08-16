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
    with open(op.template_cxx_path, "rb") as f:
        template_cxx = f.read()
else:
    template_cxx = ""

if op.template_python_path is not None and osp.isfile(op.template_python_path):
    with open(op.template_python_path, "rb") as f:
        template_python = f.read()
else:
    template_python = ""

# Read the XML
with open(op.xml_conventions_file, "rb") as f:
    xml_root = et.fromstring(f.read())

replacements = (
    ("ChestTypes/ChestType", "//##CHEST_TYPE_ENUM##"),
    ("ChestRegions/ChestRegion", "//##CHEST_REGION_ENUM##"),
    ("ImageFeatures/ImageFeature", "//##IMAGE_FEATURE_ENUM##"),
    ("ReturnCodes/ReturnCode", "//##RETURN_CODE_ENUM##")
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
)
for replacement in replacements:
    for node in xml_root.findall(replacement):
        replacement_text_cxx += "            {}.push_back( (unsigned char)( {} ) );\n".format(replacement.split("/")[0],
                                                                                              node.find("Code").text)
    replacement_text_cxx += "\n"

# Replace collections (string names)
replacement_text_cxx += "\n"
replacements = (
    "ChestTypes/ChestType",
    "ChestRegions/ChestRegion",
    "ImageFeatures/ImageFeature",
)
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
"""            // For the hierarchical relationships, leftness and rightness
            // are respected before any relationship that transcends
            // leftness or rightness. For example left lower third maps to
            // left lung, not lower third, etc. The exception to this rule
            // is that both left and right lungs are subordinate to
            // WHOLELUNG, not LEFT and RIGHT\n"""
for hierarchy_node in xml_root.findall("ChestRegionHierarchyMap/Hierarchy"):
    replacement_text_cxx += \
"""            ChestRegionHierarchyMap.insert(Region_Pair((unsigned char)({}),
                                                       (unsigned char)({}) ) );""".format(
            hierarchy_node.find("Child").text, hierarchy_node.find("Parent").text)
    replacement_text_cxx += "\n"

# Colors
i = 0
replacement_text_cxx += "\n"
replacements = (
    "ChestTypes/ChestType",
    "ChestRegions/ChestRegion")

for replacement in replacements:
    for node in xml_root.findall(replacement):
        color = node.find("Color").text.split(";")
        # replacement_text += "            double* t092 = new double[3]; t092[0] = 0.03; t092[1] = 0.03; t092[2] = 0.04; ChestTypeColors.push_back( t092 );" \
        replacement_text_cxx += "            double c{0}[] = {{{1},{2},{3}}};  {4}Colors.push_back(c{0});\n" \
            .format(i, color[0], color[1], color[2], replacement.split("/")[1])
        i += 1

    replacement_text_cxx += "\n"

template_cxx = template_cxx.replace("//##STRUCTURES##", replacement_text_cxx)

# Write output result files
if op.output_path_cxx:
    with open(op.output_path_cxx, "wb") as f:
        f.write(template_cxx)
if op.out_python_source:
    with open(op.out_python_source, "wb") as f:
        f.write(template_python)
if op.out_python_bin:
    with open(op.out_python_bin, "wb") as f:
        f.write(template_python)

print("Convention Files generated succesfully.")
if op.output_path_cxx:
    print("C++ Output path: %s" % op.output_path_cxx)
if op.out_python_source:
    print("Python source path: %s" % op.out_python_source)
if op.out_python_bin:
    print("Python dest binary folder path: %s" % op.out_python_bin)

