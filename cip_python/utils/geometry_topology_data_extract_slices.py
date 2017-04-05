import argparse
import SimpleITK as sitk

from cip_python.input_output.image_reader_writer import ImageReaderWriter
from cip_python.common.geometry_topology_data import *
from cip_python.common import ChestConventions

def extract_slices(input_volume_path, xml_input, output_dir=None, cid=None,
                   filtered_chest_regions=None, filtered_chest_types=None, num_extra_slices=0):
    """
    Extract slice/s and labelmap for all/some structures available in an xml representing a GeometryTopologyData object
    Parameters
    ----------
    input_volume_path: path to the ct
    xml_input: path to the xml file
    output_dir: folder where the will save all the results. Default: current_folder/slice_extract
    cid: base name used to name all the result volumes. Default: original name of the file
    filtered_chest_regions: comma separated list of chest regions to analyze, ignoring the rest (default: all the chest regions)
    filtered_chest_types: comma separated list of chest types to analyze, ignoring the rest (default: all the chest types)
    num_extra_slices: number of slices to take around the ground truth slice
    """
    # Read the xml
    geom = GeometryTopologyData.from_xml_file(xml_input)
    # Read the volume
    reader = ImageReaderWriter()
    vol = reader.read(input_volume_path)
    if output_dir is None:
        output_dir = os.path.join(os.path.curdir, "slice_extract")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Output folder {} created".format(os.path.realpath(output_dir)))

    if not cid:
        cid = os.path.basename(input_volume_path)

    if num_extra_slices > 0:
        # We may need to pad the image to avoid out of bounds
        pad_filter = sitk.ConstantPadImageFilter()

    for bb in geom.bounding_boxes:
        # Filtering?
        if filtered_chest_regions is not None and bb.chest_region not in filtered_chest_regions:
            continue
        if filtered_chest_types is not None and bb.chest_type not in filtered_chest_types:
            continue
        # Extract the ChestRegionType value following the CIP conventions
        chest_value = ChestConventions.GetValueFromChestRegionAndType(bb.chest_region, bb.chest_type)

        crop_filter = sitk.CropImageFilter()

        if bb.size[0] == 0:
            # Sagittal
            plane = 'S'
            slice = int(bb.start[0])    # Number of slice to extract
            total_size = vol.GetSize()[0]   # Total number of slices in this plane
            start = slice - num_extra_slices    # Cropping 'start' images "from the left" (lowerbounds)
            end = total_size - slice - num_extra_slices - 1     # Cropping 'end' images "from the right" (upperbounds)
            if end < 0:
                # Upperbounds padding needed
                # Crop the original image
                cropped_image = crop_filter.Execute(vol, [start, 0, 0], [0, 0, 0])
                # Add upperbounds padding (0s)
                cropped_image = pad_filter.Execute(cropped_image, [0,0,0], [abs(end), 0, 0], 0)
            elif start < 0:
                # Lowerbounds padding needed
                # Crop the original image
                cropped_image = crop_filter.Execute(vol, [0, 0, 0], [end, 0, 0])
                # Add lowerbounds padding (0s)
                cropped_image = pad_filter.Execute(cropped_image, [abs(start), 0, 0], [0, 0, 0], 0)
            else:
                # No padding
                cropped_image = crop_filter.Execute(vol, [start, 0, 0], [end, 0, 0])
            # Create a mask using the cropped image as a reference to keep origin, etc.
            mask = sitk.Image(cropped_image)
            mask_array = reader.sitkImage_to_numpy(mask)
            mask_array[:] = 0
            # Set the labalmap values (only ground truth bounding box)
            mask_array[num_extra_slices,
                        int(bb.start[1] + num_extra_slices):int(bb.start[1] + bb.size[1] + num_extra_slices),
                        int(bb.start[2] + num_extra_slices):int(bb.start[2] + bb.size[2] + num_extra_slices),
                      ] = 1
        elif bb.size[1] == 0:
            # Coronal
            plane = 'C'
            slice = int(bb.start[1])
            total_size = vol.GetSize()[1]
            start = slice - num_extra_slices
            end = total_size - slice - num_extra_slices - 1
            if end < 0:
                cropped_image = crop_filter.Execute(vol, [0, start, 0], [0, 0, 0])
                cropped_image = pad_filter.Execute(cropped_image, [0,0,0], [0, abs(end), 0], 0)
            elif start < 0:
                cropped_image = crop_filter.Execute(vol, [0, 0, 0], [0, end, 0])
                cropped_image = pad_filter.Execute(cropped_image, [0, abs(start), 0], [0, 0, 0], 0)
            else:
                cropped_image = crop_filter.Execute(vol, [0, start, 0], [0, end, 0])
            mask = sitk.Image(cropped_image)
            mask_array = reader.sitkImage_to_numpy(mask)
            mask_array[:] = 0
            mask_array[int(bb.start[0] + num_extra_slices):int(bb.start[0] + bb.size[0] + num_extra_slices),
                        num_extra_slices,
                        int(bb.start[2] + num_extra_slices):int(bb.start[2] + bb.size[2] + num_extra_slices),
                      ] = 1
        elif bb.size[2] == 0:
            # Axial
            plane = 'A'
            slice = int(bb.start[2])
            total_size = vol.GetSize()[2]
            start = slice - num_extra_slices
            end = total_size - slice - num_extra_slices - 1
            if end < 0:
                cropped_image = crop_filter.Execute(vol, [0, 0, start], [0, 0, 0])
                cropped_image = pad_filter.Execute(cropped_image, [0,0,0], [0, 0, abs(end)], 0)
            elif start < 0:
                cropped_image = crop_filter.Execute(vol, [0, 0, 0], [0, 0, end])
                cropped_image = pad_filter.Execute(cropped_image, [0, 0, abs(start)], [0, 0, 0], 0)
            else:
                cropped_image = crop_filter.Execute(vol, [0, 0, start], [0, 0, end])
            mask = sitk.Image(cropped_image)
            mask_array = reader.sitkImage_to_numpy(mask)
            mask_array[:] = 0
            mask_array[int(bb.start[0] + num_extra_slices):int(bb.start[0] + bb.size[0] + num_extra_slices),
                        int(bb.start[1] + num_extra_slices):int(bb.start[1] + bb.size[1] + num_extra_slices),
                        num_extra_slices] = 1
        else:
            raise Exception("Unknown plane for structure {}. Size: {}".format(bb.id), bb.size)
        # Convert the mask in a SimpleITK image back
        mask = reader.numpy_to_sitkImage(mask_array, metainfo=None, sitk_image_tempate=mask)
        # Save the file (Volume_cid_chestRegionType_plane_sliceNumber.nrrd)
        output_path = os.path.join(output_dir, "{}_{}_{}_{}.nrrd".format(cid, chest_value, plane, slice))
        sitk.WriteImage(cropped_image, output_path)
        # Save the labelmap (Volume_cid_chestRegionType_plane_sliceNumber_labelmap.nrrd)
        mask_output_path = os.path.join(output_dir, "{}_{}_{}_{}_labelmap.nrrd".format(cid, chest_value, plane, slice))
        sitk.WriteImage(mask, mask_output_path)
    print("All results saved in {}".format(os.path.realpath(output_dir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract slices and masks from structures encoded in a GeometryTopologyData xml file')
    parser.add_argument('--input', dest='input', help="Input volume path", type=str, required=False)
    parser.add_argument('--xml_input', dest='xml_input', help="Xml input path for the GeometryTopologyData object", type=str, required=False)
    parser.add_argument('--output_dir', dest='output_dir', help="Path of the output directory", type=str, required=False)
    parser.add_argument('--cid', dest='cid', help="Base name for the results", type=str, required=False)
    parser.add_argument('--num_extra_slices', dest='num_extra_slices', help="Number of extra slices surrounding each structure (it will be padded if necessary)", type=int, default=0)
    parser.add_argument('--filtered_chest_regions', dest='filtered_chest_regions', help="Comma-separated list with chest regions to filter (the rest will be ignored). If blank, all the regions will be extracted"
                        , type=str, required=False)
    parser.add_argument('--filtered_chest_types', dest='filtered_chest_types',
                        help="Comma-separated list with chest types to filter (the rest will be ignored). If blank, all the types will be extracted"
                        , type=str, required=False)
    parser.add_argument('--caselist', dest='caselist', help="Path to a caselist. The results will be saved relatively for each case",
                        type=str, required=False)

    args = parser.parse_args()
    filtered_chest_regions = map(int, args.filtered_chest_regions.split(",")) if args.filtered_chest_regions else None
    filtered_chest_types = map(int, args.filtered_chest_types.split(",")) if args.filtered_chest_types else None

    if args.caselist:
        # Open the caselist file
        with open(args.caselist, 'rb') as f:
            for case_path in map(str.strip, f.readlines()):
                xml_path = case_path.replace(".nrrd", "_structures.xml")
                print "Processing file {} ({})".format(case_path, xml_path)
                output_dir = args.output_dir if args.output_dir is not None \
                            else os.path.join(os.path.dirname(case_path), "slice_extract")
                cid = os.path.basename(case_path).replace('.nrrd', '')
                extract_slices(case_path, xml_path, output_dir=output_dir, cid=cid,
                               filtered_chest_regions=filtered_chest_regions, filtered_chest_types=filtered_chest_types,
                               num_extra_slices=args.num_extra_slices)
    else:
        # Single case
        extract_slices(args.input, args.xml_input, output_dir=args.output_dir, cid=args.cid,
                   filtered_chest_regions=filtered_chest_regions, filtered_chest_types=filtered_chest_types,
                   num_extra_slices=args.num_extra_slices)

