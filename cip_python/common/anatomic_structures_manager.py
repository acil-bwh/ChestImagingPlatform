import SimpleITK as sitk
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage.interpolation as scipy_interpolation

from cip_python.common import *
from cip_python.input_output import ImageReaderWriter

class AnatomicStructuresManager(object):
    def get_2D_numpy_from_sitk_image(self, case_path_or_sitk_volume, plane=None):
        """
        Take a 3D sitk volume and get a numpy array in "anatomical" shape.
        If the volume is 2D, the numpy array will have only 2 dimensions and the plane will be automatically deduced.
        Otherwise, the volume will have 3 dimensions but the "anatomical" view in IJK will be specified by
        the 'plane' parameter.
        Args:
            case_path_or_sitk_volume: simpleITK image or path to a file
            plane: anatomical plane (declared in CIP ChestConventions)
            new_size: 2-tuple with the width and height of the returned images

        Returns:
            2D/3D numpy array in a "natural anatomic" view.
        """
        reader = ImageReaderWriter()

        if isinstance(case_path_or_sitk_volume, sitk.Image):
            sitk_volume = case_path_or_sitk_volume
        else:
            sitk_volume = reader.read(case_path_or_sitk_volume)

        if plane is None:
            # Deduce the plane (the volume should be 2D)
            if sitk_volume.GetSize()[0] == 1:
                plane = Plane.SAGITTAL
            elif sitk_volume.GetSize()[1] == 1:
                plane = Plane.CORONAL
            elif sitk_volume.GetSize()[2] == 1:
                plane = Plane.AXIAL
            else:
                raise Exception("The volume has more than 2 dimensions and the plane has not been specified")

        arr = reader.sitkImage_to_numpy(sitk_volume)

        # Do the transformations to see the array in anatomical view
        if plane == Plane.SAGITTAL:
            arr = np.rot90(arr, axes=(1,2))
        elif plane == Plane.CORONAL:
            arr = np.rot90(arr, axes=(0, 2))
        elif plane == Plane.AXIAL:
            arr = np.flipud(np.rot90(arr, axes=(0, 1)))
        else:
            raise Exception("Wrong plane: {}".format(plane))

        # In 2D, return an array that has only 2 dimensions
        arr = arr.squeeze()

        return arr

    def get_stacked_slices_from_3D_volume(self, case_path_or_sitk_volume, plane, new_size=None):
        """
        Read a volume, and perform the required operations to get a 3D array in "anatomical view" where the FIRST
        dimension contains the slices.
        Optionally, width and height of each slice can be specified
        Args:
            case_path_or_sit
            k_volume: case_path_or_sitk_volume: simpleITK image or path to a file
            plane: Plane.AXIAL, Plane.CORONAL or Plane.SAGITTAL
            new_size: width and height of each one of the 2-D images (optional)

        Returns:
            3D numpy array. First dimension would contain the slices
        """
        reader = ImageReaderWriter()

        if isinstance(case_path_or_sitk_volume, sitk.Image):
            sitk_volume = case_path_or_sitk_volume
        else:
            sitk_volume = reader.read(case_path_or_sitk_volume)
        arr = reader.sitkImage_to_numpy(sitk_volume)

        # Do the transformations to see the array in anatomical view
        if plane == Plane.SAGITTAL:
            # Perform operations
            arr = np.rot90(arr, axes=(1, 2))
        elif plane == Plane.CORONAL:
            # Move the slices to the last dimension
            arr = np.transpose(arr, (1, 0, 2))
            # Perform operations
            arr = np.rot90(arr, k=1, axes=(1, 2))
        elif plane == Plane.AXIAL:
            # Move the slices to the first dimension
            arr = np.transpose(arr, (2, 0, 1))
            # Perform operations
            arr = np.flip(np.rot90(arr, k=3, axes=(1, 2)), axis=2)
        else:
            raise Exception("Wrong plane: {}".format(plane))

        if new_size is not None:
            # Resize (the first dimension remains intact)
            factors = [1.0, float(new_size[0]) / arr.shape[1], float(new_size[1]) / arr.shape[2]]
            arr = scipy_interpolation.zoom(arr, factors)

        return arr


    def lps_to_xywh(self, lps_coords, size, plane, lps_transformation_matrix):
        """
        Get LPS 3D coordinates that are defined in a plane, and convert it to the corresponding X,Y,W,H coordinates in
        anatomic view (top left coordinate plus width and height).
        Args:
            lps_coords: "3-D" coordinates in LPS format
            plane: plane (defined in ChestConventions)
            lps_transformation_matrix: LPS to IJK transformation matrix for the original volume

        Returns:
            list with X, Y, W, H coordinates in anatomic view
        """
        raise NotImplementedError("Suggested: First convert LPS to IJK, then call ijk_to_xywh")

    def ijk_to_xywh(self, coord1, coord2, vol_size, normalize=False):
        """
        Given coordinates and region size in IJK THAT FOLLOWS ITK CONVENTION, transform the coordinates to
        x,y,w,h that follow anatomical convention (the way an user would see the 2D images).
        If normalize==True, all the coordinates will be normalized to a 0-1 range
        Args:
            coord1: first coord in IJK
            coord2: second coord in IJK
            vol_size: tuple with volume size

        Returns:
            tuple with X,Y,W,H coordinates in an anatomical format, where X and Y represent the top-left coordinates

        """
        if coord1[0] == coord2[0]:
            # Sagittal
            xp = 1
            yp = 2
            x = int(coord1[xp])
            y = int(coord1[yp])
            width = int(abs(coord2[xp] - coord1[xp]))
            height = int(abs(coord2[yp] - coord1[yp]))
            y = vol_size[yp] - y - height
        elif coord1[1] == coord2[1]:
            # Coronal
            xp = 0
            yp = 2
            x = int(coord1[xp])
            y = int(coord1[yp])
            width = int(abs(coord2[xp] - coord1[xp]))
            height = int(abs(coord2[yp] - coord1[yp]))
            y = vol_size[yp] - y - height
        elif coord1[2] == coord2[2]:
            # Axial
            xp = 0
            yp = 1
            x = int(coord1[xp])
            y = int(coord1[yp])
            width = int(abs(coord2[xp] - coord1[xp]))
            height = int(abs(coord2[yp] - coord1[yp]))
        else:
            raise Exception("The structure is not 2D")

        if normalize:
            x /= float(vol_size[xp])
            width /= float(vol_size[xp])
            y /= float(vol_size[yp])
            height /= float(vol_size[yp])
        return x, y, width, height

    def xywh_to_3D_ijk(self, coords, num_slice, plane, volume_shape):
        """
        Convert coords xywh (where x and y are the coordinates of the CENTER of the structure) to a 3D IJK format
        that can be transferred to a GeometryTopologyData object
        Args:
            coords: np array with 4 coordinates in [0-1] range
            num_slice: int. number of slice (following the format given by "get_stacked_slices_from_3D_volume")
            plane: ChestConventions Plane object
            volume_shape: 3-tuple int with the original volume shape (in IJK, as returned by sitk_img.GetDimensions())

        Returns:
            Tuple with 2 numpy arrays (start, size) that can be copied to a GeometryTopologyData object

        """
        cs = np.copy(coords)
        if plane == Plane.AXIAL:
            cs[:2] *= volume_shape[:2]
            cs[2:] *= volume_shape[:2]
            xmin = max(int(cs[0] - (cs[2] / 2)), 0)
            ymin = max(int(cs[1] - (cs[3] / 2)), 0)
            start = np.array([xmin, ymin, num_slice])
            size = np.array([int(cs[2]), int(cs[3]), 0])
        elif plane == Plane.SAGITTAL:
            cx = 1
            cy = 2
            cs = np.clip(cs, 0, 0.999)
            cs[0] *= volume_shape[cx]
            cs[1] *= volume_shape[cy]
            cs[2] *= volume_shape[cx]
            cs[3] *= volume_shape[cy]
            xmin = max(int(cs[0] - (cs[2] / 2)), 0)
            ymin = max(volume_shape[cy] - int(cs[1] + (cs[3] / 2)), 0)
            start = np.array([num_slice, xmin, ymin])
            size = np.array([0, int(cs[2]), int(cs[3])])
        elif plane == Plane.CORONAL:
            cx = 0
            cy = 2
            cs = np.clip(cs, 0, 0.999)
            cs[0] *= volume_shape[cx]
            cs[1] *= volume_shape[cy]
            cs[2] *= volume_shape[cx]
            cs[3] *= volume_shape[cy]

            xmin = max(int(cs[0] - (cs[2] / 2)), 0)
            ymin = max(volume_shape[cy] - int(cs[1] + (cs[3] / 2)), 0)
            start = np.array([xmin, num_slice, ymin])
            size = np.array([int(cs[2]), 0, int(cs[3])])
        else:
            raise Exception("Wrong plane: {}".format(plane))

        return start, size

    def generate_all_slices(self, case_path, output_folder):
        """ DEPRECATED.
        Generate and save all the slices of a 3D volume (SimpleITK image) in a particular plane
        Args:
           case_path: path to the CT volume
           output_folder: output folder where the images will be stored. Every slice will have 3 digits.
        """

        sitk_volume = ImageReaderWriter().read(case_path)
        case = os.path.basename(case_path).replace(".nrrd","")

        # Sagittal
        p = "{}/{}/{}".format(output_folder, case, ChestConventions.GetPlaneName(Plane.SAGITTAL))
        if not os.path.isdir(p):
            os.makedirs(p)
        for i in range(sitk_volume.GetSize()[0]):
            sitk.WriteImage(sitk_volume[i:i + 1, :, :], "{}/{:03}.nrrd".format(p, i))

        # Coronal
        p = "{}/{}/{}".format(output_folder, case, ChestConventions.GetPlaneName(Plane.CORONAL))
        if not os.path.isdir(p):
            os.makedirs(p)
        for i in range(sitk_volume.GetSize()[1]):
            sitk.WriteImage(sitk_volume[:, i:i + 1, :], "{}/{:03}.nrrd".format(p, i))

        # Axial
        p = "{}/{}/{}".format(output_folder, case, ChestConventions.GetPlaneName(Plane.AXIAL))
        if not os.path.isdir(p):
            os.makedirs(p)
        for i in range(sitk_volume.GetSize()[2]):
            sitk.WriteImage(sitk_volume[:, :, i:i + 1], "{}/{:03}.nrrd".format(p, i))

    def get_cropped_structure(self, case_path_or_sitk_volume, xml_file_path, region, plane,
                              extra_margin=None, padding_constant_value=0):
        """
        Get a simpleitk volume with the bounding box content of the specified region-plane in a CT volume.
        Args:
            case_path_or_sitk_volume: path to the CT volume or sitk Image read with CIP ImageReaderWriter
            xml_file_path: Full path to a GeometryTopologyObject XML file
            region: CIP ChestRegion value
            plane: CIP Plane value
            extra_margin: 3-tuple that contains the extra margin (in pixels) for each dimension where the user wants
                           to expand the bounding box. Note that we are using ITK convention, which means:
                           0=sagittal, 1=coronal, 2=axial.
                           When -1 is used, all the slices in that plane will be used
            padding_constant_value: value used in case the result volume has to be padded because of the position of
                                    the structure and the provided spacing

        Returns:
            Sitk volume
        """
        if extra_margin is None:
            extra_margin = [0, 0, 0]
        gtd = GeometryTopologyData.from_xml_file(xml_file_path)
        if isinstance(case_path_or_sitk_volume, sitk.Image):
            sitk_volume = case_path_or_sitk_volume
        else:
            sitk_volume = ImageReaderWriter().read(case_path_or_sitk_volume)
        pad_filter = sitk.ConstantPadImageFilter()
        structure_code = ChestConventions.GetChestRegionName(region) + ChestConventions.GetPlaneName(plane)
        for bb in gtd.bounding_boxes:
            if bb.description.upper() == structure_code.upper():
                start = [0, 0, 0]
                end = [0, 0, 0]
                padding_in = [0, 0, 0]
                padding_out = [0, 0, 0]
                for i in range(3):
                    # If margin == -1 we will take the full slice
                    if extra_margin[i] == -1:
                        # Take full slice
                        start[i] = 0
                        end[i] = sitk_volume.GetSize()[i]
                    else:
                        # Crop the structure
                        start[i] = int(bb.start[i]) - extra_margin[i]
                        end[i] = int(bb.start[i] + bb.size[i]) + extra_margin[i] + 1
                        if start[i] < 0:
                            padding_in[i] = abs(start[i])
                            start[i] = 0
                        if end[i] >= sitk_volume.GetSize()[i]:
                            padding_out[i] = end[i] - sitk_volume.GetSize()[i]
                            end[i] = sitk_volume.GetSize()[i] - 1
                        if start[i] == end[i]:
                            end[i] += 1

                # Crop the image
                im = sitk_volume[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                im = pad_filter.Execute(im, padding_in, padding_out, padding_constant_value)
                return im

        # If the execution reaches this line, the structure was not found
        raise Exception("Structure {} not found in {}".format(structure_code, xml_file_path))

    def get_full_slice(self, case_path_or_sitk_volume, xml_file_path, region, plane):
        """
        Extract a sitk 3D volume that contains the structure provided
        Args:
            case_path_or_sitk_volume: path to the CT volume or sitk Image read with CIP ImageReaderWriter
            xml_file_path: Full path to a GeometryTopologyObject XML file
            region: CIP ChestRegion
            plane: CIP Plane
        Returns:
            Simple ITK image. It will be a 3D volume but one dimension will have a size=1
        """
        if plane == Plane.SAGITTAL:
            margin = [0, -1, -1]
        elif plane == Plane.CORONAL:
            margin = [-1, 0, -1]
        elif plane == Plane.AXIAL:
            margin = [-1, -1, 0]

        return self.get_cropped_structure(case_path_or_sitk_volume, xml_file_path, region, plane, extra_margin=margin)

    def generate_qc_images(self, case_path_or_sitk_volume, xml_file_path, output_folder, structures=None,
                           rectangle_color='r', line_width=3):
        """
        Generate all the png full-slice images and a red rectangle with the structure detection
        Args:
            case_path_or_sitk_volume: path to the CT volume or sitk Image read with CIP ImageReaderWriter
            xml_file_path: file path to the XML that contains the GeometryTopologyData object
            output_folder: output folder path where all the png images will be saved
            structures: when filled, a list with the structure codes that we want to generate
            rectangle_color: color for matplotlib rectangle
            line_width: line width for matplotlib rectangle
        """
        gtd = GeometryTopologyData.from_xml_file(xml_file_path)
        if isinstance(case_path_or_sitk_volume, sitk.Image):
            sitk_vol = case_path_or_sitk_volume
        else:
            sitk_vol = ImageReaderWriter().read(case_path_or_sitk_volume)

        if not osp.exists(output_folder):
            os.makedirs(output_folder)

        # Create a png for every structure
        for bb in gtd.bounding_boxes:
            if structures is None or bb.description in structures:
                if bb.description.endswith("Sagittal"):
                    slice_sitk = sitk_vol[int(bb.start[0]):int(bb.start[0]) + 1, :, :]
                    plane = Plane.SAGITTAL
                elif bb.description.endswith("Coronal"):
                    slice_sitk = sitk_vol[:, int(bb.start[1]):int(bb.start[1]) + 1, :]
                    plane = Plane.CORONAL
                elif bb.description.endswith("Axial"):
                    slice_sitk = sitk_vol[:, :, int(bb.start[2]):int(bb.start[2]) + 1]
                    plane = Plane.AXIAL
                else:
                    raise Exception("Wrong structure: {}-{}".format(bb.id, bb.description))

                slice_np = self.get_2D_numpy_from_sitk_image(slice_sitk, plane=plane)
                x, y, width, height = self.ijk_to_xywh(bb.start, bb.coord2, sitk_vol.GetSize())

                # Draw rectangle
                fig, axes = plt.subplots(nrows=1, ncols=1)
                axes.imshow(slice_np, cmap='gray')
                rect = patches.Rectangle((x, y), width, height, linewidth=line_width, edgecolor=rectangle_color,
                                         facecolor='none')
                axes.add_patch(rect)
                axes.axis('off')

                # Save the image
                name = "{}_{}.png".format(osp.basename(xml_file_path), bb.description)
                fig.savefig(osp.join(output_folder, name), bbox_inches='tight')
                plt.close()
                print (name + " generated")

        print ("Case {} finished".format(xml_file_path))

    def qc_structure_with_gt(self, case_path_or_sitk_volume, xml_file_path_pred, xml_file_path_gt, structure, output_file,
                           rectangle_color_pred='r', rectangle_color_gt='b', line_width=2):
        """
        Generate a 2D QC image comparing a structure with the ground truth.
        The slice will be the predicted one, and the ground truth will be proyected in that slice (in a dashed line)
        Args:
            case_path_or_sitk_volume: sitk volume or path to the nrrd file
            xml_file_path_pred: XML path for the predictions
            xml_file_path_gt: XML path for the ground truth
            structure: structure code
            output_file: path to the output file
            rectangle_color_pred: color for the prediction bounding box
            rectangle_color_gt: color for the ground truth bounding box
            line_width: bounding boxes width

        Returns:

        """
        if isinstance(case_path_or_sitk_volume, sitk.Image):
            imsitk = case_path_or_sitk_volume
        else:
            imsitk = ImageReaderWriter().read(case_path_or_sitk_volume)

        gtd_pred = GeometryTopologyData.from_xml_file(xml_file_path_pred)
        gtd_gt = GeometryTopologyData.from_xml_file(xml_file_path_gt)

        start_pred = size_pred = None
        for bb in gtd_pred.bounding_boxes:
            if bb.description == structure:
                start_pred = np.array(bb.start, dtype=int)
                size_pred = np.array(bb.size, dtype=int)
                break
        assert start_pred is not None, "Structure not found"

        start_gt = size_gt = None
        for bb in gtd_gt.bounding_boxes:
            if bb.description == structure:
                start_gt = np.array(bb.start, dtype=int)
                size_gt = np.array(bb.size, dtype=int)
                break
        assert start_gt is not None, "Structure not found"

        if size_gt[0] == 0:
            plane = Plane.SAGITTAL
            a = self.get_2D_numpy_from_sitk_image(imsitk, plane=plane)
            slice_img = a[start_pred[0], :, :]
        elif size_gt[1] == 0:
            plane = Plane.CORONAL
            a = self.get_2D_numpy_from_sitk_image(imsitk, plane=plane)
            slice_img = a[:, start_pred[1], :]
        elif size_gt[2] == 0:
            plane = Plane.AXIAL
            a = self.get_2D_numpy_from_sitk_image(imsitk, plane=plane)
            slice_img = a[:, :, start_pred[2]]
        else:
            raise Exception("Plane could not be inferred")

        fig, axis = plt.subplots(nrows=1)
        plt.imshow(slice_img, cmap='gray')

        # Prediction
        coord1 = start_pred
        coord2 = coord1 + size_pred
        coords = self.ijk_to_xywh(coord1, coord2, imsitk.GetSize())
        rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=line_width, edgecolor=rectangle_color_pred,
                                 facecolor='none')
        axis.add_patch(rect)

        # Ground truth
        coord1 = start_gt
        coord2 = coord1 + size_gt
        coords = self.ijk_to_xywh(coord1, coord2, imsitk.GetSize())
        rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=line_width, edgecolor=rectangle_color_gt,
                                 facecolor='none',
                                 linestyle='dashed')
        axis.add_patch(rect)
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(output_file)
        print(output_file, " saved")

    def numpy_to_sitk(self, array_2D, plane, lps_transformation_matrix):
        """
        Get a sitk 3D image from a 2D numpy array, using the corresponding transformations
        Args:
            array_2D:
            plane:
            lps_transformation_matrix:

        Returns:
            sitk 3D volume

        """
        raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate QC images from GeometryTopologyData XML files')

    parser.add_argument('operation', help="Operation", type=str,
                        choices=['generate_qc_images', 'extract_slices', 'generate_all_slices_case'])
    parser.add_argument('--case_path',  help="Case file (full path)", type=str, required=True)
    parser.add_argument('--xml_path', help="GeometryTopologyData XML file (full path)", type=str)
    parser.add_argument('--output_folder', '-o', help="Output figures folder", type=str, required=True)

    parser.add_argument('--chest_regions_planes', '-crp', type=str, required=False, nargs='*',
                        help="List of tuples that contain chest regions and planes. Ex: DESCENDINGAORTA-SAGITTAL, STERNUM-AXIAL")
    parser.add_argument('--extra_margin', '-m', type=int, nargs=3, help='Margin in each plane')

    args = parser.parse_args()
    gen = AnatomicStructuresManager()
    if args.operation == 'generate_qc_images':
        gen.generate_qc_images(args.case_path, args.xml_path, args.output_folder)
    elif args.operation == 'extract_slices':
        writer = ImageReaderWriter()
        for pair in args.chest_regions_planes:
            # Get the codes for the Chest Regions and Planes
            structure = pair.split('-')
            region = ChestConventions.GetChestRegionValueFromName(structure[0])
            plane = ChestConventions.GetPlaneValueFromName(structure[1])
            sitk_image = gen.get_cropped_structure(args.case_path, args.xml_path, region, plane, args.extra_margin)
            # Write the result as nrrd file
            case_id = os.path.basename(args.case_path).replace('.nrrd', '')
            file_name = os.path.join(args.output_folder, "{}-{}_{}.nrrd".format(case_id, structure[0], structure[1]))
            writer.write(sitk_image, os.path.join(args.output_folder, file_name))
            print ("{} generated".format(file_name))
    elif args.operation == 'generate_all_slices_case':
        gen.generate_all_slices(args.case_path, args.output_folder)
