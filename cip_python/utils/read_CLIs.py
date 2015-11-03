""" Iterate over the CommandLineTools folder and gets a tab separated values file with the following colums:
- Category
- Name
- Description
"""

import os
import sys
import os.path as path
import xml.etree.ElementTree as et

currentpath = os.path.dirname(path.realpath(__file__))
clis_path = path.normpath(path.join(currentpath, "..", "..", "CommandLineTools"))
if len(sys.argv) == 1:
    output_path = "cliResults.tsv"
else:
    output_path = sys.argv[1]

with open(output_path, "w+b") as output:
    for cli_name in os.listdir(clis_path):
        cli = path.join(clis_path, cli_name)
        if path.isdir(cli):
            xmlfile = path.join(cli, cli_name + ".xml")
            if os.path.isfile(xmlfile):
                with open(xmlfile, 'r+b') as input:
                    xml = input.read()
                    root = et.fromstring(xml)
                    category = root.find("category").text
                    category = category.split(".")[-1]
                    text = root.find("description").text
                    text = text.replace("\\", "")
                    text = " ".join(text.split())
                    output.write("{0}\t{1}\t{2}\n".format(category, cli_name, text))
print("Results saved in " + path.realpath(output_path))