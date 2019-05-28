import warnings

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

    def get_cropped_structure(self, case_path_or_sitk_volume, xml_file_path_or_GTD_object, region, plane,
                              extra_margin=None, padding_constant_value=0, out_of_bouds_tolerance_pixels=0):
        """
        Get a simpleitk volume with the bounding box content of the specified region-plane in a CT volume.
        Args:
            case_path_or_sitk_volume: str. Path to the CT volume or sitk Image read with CIP ImageReaderWriter
            xml_file_path_or_GTD_object: str. Full path to a GeometryTopologyObject XML file or the object itself
            region: int. CIP ChestRegion value
            plane: int. CIP Plane value
            extra_margin: 3-int-tuple that contains the extra margin (in pixels) for each dimension where the user wants
                           to expand the bounding box. Note that we are using ITK convention, which means:
                           0=sagittal, 1=coronal, 2=axial.
                           When -1 is used, all the slices in that plane will be used
            padding_constant_value: int. Value used in case the result volume has to be padded because of the position of
                                    the structure and the provided spacing
        Returns:
            Sitk volume with the cropped structure or None if it was not found
        """
        gtd = GeometryTopologyData.from_xml_file(xml_file_path_or_GTD_object) if isinstance(xml_file_path_or_GTD_object, str) \
              else xml_file_path_or_GTD_object

        if isinstance(case_path_or_sitk_volume, sitk.Image):
            sitk_volume = case_path_or_sitk_volume
        else:
            sitk_volume = ImageReaderWriter().read(case_path_or_sitk_volume)

        if extra_margin is None:
            extra_margin = [0, 0, 0]

        pad_filter = sitk.ConstantPadImageFilter()
        structure_code = ChestConventions.GetChestRegionName(region) + ChestConventions.GetPlaneName(plane)
        if plane == Plane.AXIAL:
            slix = 2
        elif plane == Plane.CORONAL:
            slix = 1
        elif plane == Plane.SAGITTAL:
            slix = 0
        else:
            raise Exception("Wrong plane")
        for bb in gtd.bounding_boxes:
            if bb.description.upper() == structure_code.upper():
                start = [0, 0, 0]
                end = [0, 0, 0]
                padding_in = [0, 0, 0]
                padding_out = [0, 0, 0]

                # Check if the slice is inbounds. Otherwise return None
                if -start[slix] > out_of_bouds_tolerance_pixels or \
                        start[slix] >= (out_of_bouds_tolerance_pixels + sitk_volume.GetSize()[slix]):
                    # Structure plane out of bounds
                    return None
                else:
                    # Clip the structure plane coordinate
                    start[slix] = np.clip(start[slix], 0, sitk_volume.GetSize()[slix] - 1)

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
        return None

    def get_full_slice(self, case_path_or_sitk_volume, xml_file_path_or_GTD_object, region, plane, out_of_bounds_slice_tolerance_pixels=0):
        """
        Extract a sitk 3D volume that contains the structure provided
        Args:
            case_path_or_sitk_volume: path to the CT volume or sitk Image read with CIP ImageReaderWriter
            xml_file_path_or_GTD_object: Full path to a GeometryTopologyObject XML file or the object itself
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
        else:
            raise Exception("Wrong plane: {}".format(plane))

        return self.get_cropped_structure(case_path_or_sitk_volume, xml_file_path_or_GTD_object, region, plane,
                                          extra_margin=margin, out_of_bouds_tolerance_pixels=out_of_bounds_slice_tolerance_pixels)

    def get_structure_coordinates(self, xml_file_path_or_GTD_object, region, plane, dtype=np.int32):
        """
        Gets the coordinates of the bounding box content of the specified region-plane in a CT volume.
        :param xml_file_path_or_GTD_object: Full path to a GeometryTopologyObject XML file or the object itself
        :param region: CIP ChestRegion value
        :param plane: CIP Plane value
        :return: tuple with 2 numpy arrays with start and size coordinates
        """
        gtd = GeometryTopologyData.from_xml_file(xml_file_path_or_GTD_object) \
            if isinstance(xml_file_path_or_GTD_object, str) else xml_file_path_or_GTD_object

        structure_code = ChestConventions.GetChestRegionName(region) + ChestConventions.GetPlaneName(plane)
        for bb in gtd.bounding_boxes:
            if bb.description.upper() == structure_code.upper():
                bb.convert_to_array(dtype)
                return bb.start, bb.size

        # If the execution reaches this line, the structure was not found
        return None

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
                try:
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
                except Exception as ex:
                    print ("Error in structure {}: {}".format(bb.description, ex))

        print ("Case {} finished".format(xml_file_path))

    def qc_structure_with_gt(self, case_path_or_sitk_volume, xml_file_path_pred, xml_file_path_gt,
                             structures=None, output_folder=None, output_file_type="png",
                             fig_size=(10,10),
                             rectangle_color_pred='r', rectangle_color_gt='b',
                             rectangle_color_out_of_bounds='yellow',
                             line_width=2, plot_inline=False):
        """
        Generate a 2D QC image comparing a structure with the ground truth.
        The slice will be the predicted one, and the ground truth will be projected in that slice (in a dashed line).
        By default, the prediction will be in a red solid line, while the ground truth projection will be a blue dashed line.
        If the prediction is out of bounds, the closest slice will be chosen (first or last one), and the bounding box
        will be a solid yellow line
        Args:
            case_path_or_sitk_volume: sitk volume or path to the nrrd file
            xml_file_path_pred: XML path for the predictions
            xml_file_path_gt: XML path for the ground truth
            structures: list/set/tuple of structure codes to be analyzed. If None, all the structures will be analyzed
            output_folder: path to the output folder where the files will be stored. If None, the figures won't be stored
                           (just plotted)
            output_file_type: str. Extension of the saved image
            fig_size: 2-int tuple: Fig size (default: (10,10))
            rectangle_color_pred: color for the prediction bounding box
            rectangle_color_gt: color for the ground truth bounding box
            rectangle_color_out_of_bounds: color for the out of bounds bounding box predictions
            line_width: bounding boxes width
            plot_inline: when True, the images will be plotted inline too. Otherwise, just save the figure
        """
        if isinstance(case_path_or_sitk_volume, sitk.Image):
            imsitk = case_path_or_sitk_volume
        else:
            imsitk = ImageReaderWriter().read(case_path_or_sitk_volume)

        gtd_pred = GeometryTopologyData.from_xml_file(xml_file_path_pred)
        gtd_gt = GeometryTopologyData.from_xml_file(xml_file_path_gt)

        if structures is not None:
            structures = set(structures)

        predictions = {}
        all_strs = set()

        # Read all the predictions
        for bb in (bb for bb in gtd_pred.bounding_boxes if (structures is None or bb.description in structures)):
            start = np.array(bb.start, dtype=int)
            size = np.array(bb.size, dtype=int)
            predictions[bb.description] = (start, size)
            all_strs.add(bb.description)

        # Read all the ground truths
        gts = {}
        for bb in (bb for bb in gtd_gt.bounding_boxes if (structures is None or bb.description in structures)):
            start = np.array(bb.start, dtype=int)
            size = np.array(bb.size, dtype=int)
            gts[bb.description] = (start, size)

        # Draw each one of the structures
        for str in all_strs:
            # Get the predicted slice
            start = predictions[str][0]
            color = rectangle_color_pred
            if str.endswith('Sagittal'):
                a = self.get_2D_numpy_from_sitk_image(imsitk, plane=Plane.SAGITTAL)
                slice_ix = start[0]
                if slice_ix < 0:
                    slice_ix = 0
                    color = rectangle_color_out_of_bounds
                elif slice_ix >= a.shape[0]:
                    slice_ix = a.shape[0] - 1
                    color = rectangle_color_out_of_bounds
                slice_img = a[slice_ix, :, :]
            elif str.endswith("Coronal"):
                a = self.get_2D_numpy_from_sitk_image(imsitk, plane=Plane.CORONAL)
                slice_ix = start[1]
                if slice_ix < 0:
                    slice_ix = 0
                    color = rectangle_color_out_of_bounds
                elif slice_ix >= a.shape[1]:
                    slice_ix = a.shape[1] - 1
                    color = rectangle_color_out_of_bounds
                slice_img = a[:, slice_ix, :]
            elif str.endswith("Axial"):
                a = self.get_2D_numpy_from_sitk_image(imsitk, plane=Plane.AXIAL)
                slice_ix = start[2]
                if slice_ix < 0:
                    slice_ix = 0
                    color = rectangle_color_out_of_bounds
                elif slice_ix >= a.shape[2]:
                    slice_ix = a.shape[2] - 1
                    color = rectangle_color_out_of_bounds
                slice_img = a[:, :, slice_ix]
            else:
                raise Exception("Plane could not be inferred")

            # Plot image
            fig = plt.figure(figsize=fig_size)
            axis = plt.Axes(fig, [0., 0., 1., 1.])
            axis.set_axis_off()
            fig.add_axes(axis)
            axis.imshow(slice_img, cmap='gray')

            # Prediction
            coord1 = start
            coord2 = coord1 + predictions[str][1]
            coords = self.ijk_to_xywh(coord1, coord2, imsitk.GetSize())
            rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=line_width, edgecolor=color,
                                     facecolor='none')
            axis.add_patch(rect)

            if str in gts:
                # Ground truth
                coord1 = gts[str][0]
                coord2 = coord1 + gts[str][1]
                coords = self.ijk_to_xywh(coord1, coord2, imsitk.GetSize())
                rect = patches.Rectangle((coords[0], coords[1]), coords[2], coords[3], linewidth=line_width, edgecolor=rectangle_color_gt,
                                         facecolor='none',
                                         linestyle='dashed')
                axis.add_patch(rect)

            if output_folder is not None:
                # Save figure in file
                os.makedirs(output_folder, exist_ok=True)
                output_file = "{}/{}_{}.{}".format(output_folder,
                                                    os.path.basename(xml_file_path_gt).replace("_structures.xml", ""),
                                                    str,
                                                    output_file_type)
                plt.savefig(output_file)
                print(output_file, " saved")
            if not plot_inline:
                # Close the figure
                if output_folder is None:
                    warnings.warn("No figures will be saved or plotted. Please specify an output folder to save the figures")
                plt.close()

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
