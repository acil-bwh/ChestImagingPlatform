""" Iterate over the CommandLineTools folder and gets a tab separated values file with the following colums:
- Category
- Name
- Description
"""

import os
import os.path as path
import xml.etree.ElementTree as et

def generate_CLIs_description():
    currentpath = os.path.dirname(path.realpath(__file__))
    clis_path = path.normpath(path.join(currentpath, "..", "..", "CommandLineTools"))
    output = ""

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
                    output += "{0}\t{1}\t{2}\n".format(category, cli_name, text)
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract CLIs info via their XML description files.')
    parser.add_argument("--out", dest="output_file", required=True, help='Output results txt file')
    options = parser.parse_args()
    clis = generate_CLIs_description()

    with open(options.output_file, "w+b") as output:
        output.write(clis)

    print("Results saved successfully in " + options.output_file)