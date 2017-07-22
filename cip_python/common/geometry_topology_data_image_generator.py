# Extract regions and generate quality control images from structures located in a GeometryTopologyData object
import argparse
import os.path as osp
import numpy as np

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from acil_python.data.geometry_topology_data import *
import acil_python.data.archive_manager_facade as facade
from cip_python.input_output import ImageReaderWriter

class GeometryTopologyDataImageGenerator(object):
    """ Extract regions and generate quality control images from structures located in a GeometryTopologyData object"""

    def generate_images(self, case_path, xml_file_path, output_folder, rectangle_color='r', line_width=3):
        """
        Generate all the images and a red rectangle with the structure detection
        Args:
            case_path: file path to the CT volume
            xml_file_path: file path to the XML that contains the GeometryTopologyData object
            output_folder: output folder path where all the png images will be saved
        """
        gtd = GeometryTopologyData.from_xml_file(xml_file_path)
        ct_array = ImageReaderWriter().read_in_numpy(case_path)[0]
        case = xml_file_path.split('/')[-1].replace('.xml', '')

        if not osp.exists(output_folder):
            os.makedirs(output_folder)

        # Create a png for every structure
        for bb in gtd.bounding_boxes:
            if bb.description.endswith('Axial'):
                xp = 0
                yp = 1
                zp = 2
                x = int(bb.start[xp])
                y = int(bb.start[yp])
                z = int(bb.start[zp])
                width = int(bb.size[xp])
                height = int(bb.size[yp])
                im = ct_array[:, :, z]
                im = np.flipud(np.rot90(im))
            elif bb.description.endswith('Sagittal'):
                xp = 1
                yp = 2
                zp = 0
                x = int(bb.start[xp])
                y = int(bb.start[yp])
                z = int(bb.start[zp])
                width = int(bb.size[xp])
                height = int(bb.size[yp])
                y = ct_array.shape[yp] - y - height
                im = ct_array[z, :, :]
                im = np.rot90(im)
            elif bb.description.endswith('Coronal'):
                xp = 0
                yp = 2
                zp = 1
                x = int(bb.start[xp])
                y = int(bb.start[yp])
                z = int(bb.start[zp])
                width = int(bb.size[xp])
                height = int(bb.size[yp])
                y = ct_array.shape[yp] - y - height
                im = ct_array[:, z, :]
                im = np.rot90(im)
            else:
                raise Exception("Wrong plane for structure " + bb.description)

            # Draw rectangle
            fig, axes = plt.subplots(nrows=1, ncols=1)
            axes.imshow(im, cmap='gray')
            rect = patches.Rectangle((x, y), width, height, linewidth=line_width, edgecolor=rectangle_color, facecolor='none')
            axes.add_patch(rect)
            axes.axis('off')

            # Save the image
            name = "{}_{}.png".format(case, bb.description)
            fig.savefig(osp.join(output_folder, name), bbox_inches='tight')
            plt.close()
            print (name + " generated")

        print ("Case {} finished".format(xml_file_path))

    def get_cropped_detection(self, case_path, xml_file_path, structure_code_list):
        """
        Get a list of numpy arrays with the bounding box content of the specified structures in a CT volume.
        The xml file path should specify a path to a GeometryTopologyData object
        Args:
            case_path: path to the CT volume
            xml_file_path: Full path to a GeometryTopologyObject XML file
            structure_code_list: list of structure codes (text). Example: [WholeHeartAxial, DescendingAortaSagittal]

        Returns:
            List of Numpy arrays in the format that the user is used to see the images ("anatomical" way)
        """

        gtd = GeometryTopologyData.from_xml_file(xml_file_path)
        case_array = ImageReaderWriter().read_in_numpy(case_path)[0]
        result = []
        for structure_code in structure_code_list:
            for bb in gtd.bounding_boxes:
                if bb.description == structure_code:
                    # Get the cropped image
                    if bb.size[0] == 0:
                        # Sagittal
                        cropped_image = case_array[ int(bb.start[0]),
                                                    int(bb.start[1]):int(bb.start[1] + bb.size[1]),
                                                    int(bb.start[2]):int(bb.start[2] + bb.size[2])]
                        # Show as an user would expect
                        cropped_image = np.rot90(cropped_image)

                    elif bb.size[1] == 0:
                        # Coronal
                        cropped_image = case_array[ int(bb.start[0]):int(bb.start[0] + bb.size[0]),
                                                    int(bb.start[1]),
                                                    int(bb.start[2]):int(bb.start[2] + bb.size[2])]
                        # Show as an user would expect
                        cropped_image = np.rot90(cropped_image)
                    else:
                        # Axial
                        cropped_image = case_array[ int(bb.start[0]):int(bb.start[0] + bb.size[0]),
                                                    int(bb.start[1]):int(bb.start[1] + bb.size[1]),
                                                    int(bb.start[2])]
                        # Show as an user would expect
                        cropped_image = np.flipud(np.rot90(cropped_image))

                    result.append(cropped_image)
                    # Stop searching for another structures
                    break

            # If the execution reaches this line, the structure was not found
            raise Exception("Structure {} not found in {}".format(structure_code, xml_file_path))

        # Return the list of arrays
        return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QC images from GeometryTopologyData XML files')

    parser.add_argument('case_path',  help="Case file (full path)", type=str)
    parser.add_argument('xml_path', help="GeometryTopologyData XML file (full path)", type=str)
    parser.add_argument('output_folder', help="Output figures folder", type=str)

    args = parser.parse_args()
    gen = GeometryTopologyDataImageGenerator()
    gen.generate_images(args.case_path, args.xml_path, args.output_folder)