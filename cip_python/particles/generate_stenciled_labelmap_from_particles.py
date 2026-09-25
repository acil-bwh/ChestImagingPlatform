import sys
import argparse
import vtk
import numpy as np
import math

from cip_python.input_output.image_reader_writer import ImageReaderWriter


class StenciledLabelMap:
    def __init__(self):
        self.io = ImageReaderWriter()
        self.UNDEFINEDREGION = 0
        # CIP ChestType enum values
        self.AIRWAY_TYPE = 2
        self.VESSEL_TYPE = 3
        self.FISSURE_TYPE = 55

    @staticmethod
    def get_value_from_chest_region_and_type(region, chest_type):
        return int(region) + (int(chest_type) << 8)

    @staticmethod
    def get_chest_type_from_value(value):
        return int(value) >> 8

    def execute(self, ip, ilm, output_path, airway=False, vessel=False, fissure=False,
                sigma=0.0, radius=1.0, height=1.0, cylinder=False, sphere=False,
                scale=False, dnn_radius=False, dnn_radius_name=None):

        if not cylinder and not sphere:
            print("  [ERROR] Must specify to use either --cylinder or --sphere stencil")
            sys.exit(1)

        if airway:
            hevec_name = "hevec2"
            default_chest_type = self.AIRWAY_TYPE
        elif vessel:
            hevec_name = "hevec0"
            default_chest_type = self.VESSEL_TYPE
        elif fissure:
            hevec_name = "hevec1"
            default_chest_type = self.FISSURE_TYPE
        else:
            print("  [ERROR] Must specify structure type (--airway, --vessel, or --fissure)")
            sys.exit(1)

        if dnn_radius and not dnn_radius_name:
            print("  [ERROR] When setting --dnn_radius, --dnn_radius_name must be specified")
            sys.exit(1)

        print("Reading label map...")

        sitk_image = self.io.read(ilm)
        np_image, metainfo = self.io.read_in_numpy(ilm)

        size = sitk_image.GetSize()
        origin = sitk_image.GetOrigin()
        spacing = sitk_image.GetSpacing()
        direction = sitk_image.GetDirection()

        print("Reading particles...")

        polyReader = vtk.vtkPolyDataReader()
        polyReader.SetFileName(ip)
        polyReader.Update()
        polyData = polyReader.GetOutput()

        points = polyData.GetPoints()
        num_points = points.GetNumberOfPoints()

        pointData = polyData.GetPointData()
        chestRegionArray = pointData.GetArray("ChestRegionChestType")

        scaleArray = pointData.GetArray("scale") if scale else None

        dnnRadiusArray = None
        if dnn_radius:
            if not dnn_radius_name:
                print("  [ERROR] When setting --dnn_radius, --dnn_radius_name must be specified")
                sys.exit(1)
            dnnRadiusArray = pointData.GetArray(dnn_radius_name)
            if dnnRadiusArray is None:
                print(f"  [ERROR] Array '{dnn_radius_name}' not found in particle point data.")
                sys.exit(1)

        hevecArray = pointData.GetArray(hevec_name)
        CTPointSpreadFunctionSigma = sigma

        direction_mat = np.array(direction).reshape(3, 3)
        spacing_mat = np.diag(spacing)
        affine = direction_mat @ spacing_mat
        affine_inv = np.linalg.inv(affine)
        origin_arr = np.array(origin)

        print("Creating stenciled label map...")

        out_lm = np.zeros(np_image.shape, dtype=np.uint16)
        for i in range(num_points):
            pt = points.GetPoint(i)

            if chestRegionArray is not None:
                typeValue = self.get_chest_type_from_value(chestRegionArray.GetTuple1(i))
            else:
                print("  [WARNING] Array ChestRegionChestType not found in particle point data. "
                      "Using default type values.")
                typeValue = default_chest_type
            foregroundLabel = self.get_value_from_chest_region_and_type(self.UNDEFINEDREGION, typeValue)

            center = np.array(pt)

            orientation = np.array([0.0, 0.0, 1.0])
            if hevecArray is not None:
                orientation = np.array(hevecArray.GetTuple3(i))

            current_radius = radius
            if airway or vessel:  # scale and DNN radius scaling are only applied to airway and vessel
                if scale and scaleArray is not None:
                    particle_scale = scaleArray.GetTuple1(i)
                    current_radius = math.sqrt(particle_scale ** 2 + CTPointSpreadFunctionSigma ** 2)

                if dnn_radius and dnnRadiusArray is not None:
                    current_radius = dnnRadiusArray.GetTuple1(i)

            # Determine bounding box
            if sphere:
                bbMin = center - current_radius
                bbMax = center + current_radius
            elif cylinder:
                max_ext = current_radius + height / 2.0
                bbMin = center - max_ext
                bbMax = center + max_ext

            corners = np.array([
                [bbMin[0], bbMin[1], bbMin[2]],
                [bbMin[0], bbMin[1], bbMax[2]],
                [bbMin[0], bbMax[1], bbMin[2]],
                [bbMin[0], bbMax[1], bbMax[2]],
                [bbMax[0], bbMin[1], bbMin[2]],
                [bbMax[0], bbMin[1], bbMax[2]],
                [bbMax[0], bbMax[1], bbMin[2]],
                [bbMax[0], bbMax[1], bbMax[2]]
            ])

            # Transform bounding box corners to image indices to find the extent
            indices = (corners - origin_arr) @ affine_inv.T
            indices = np.round(indices).astype(int)

            min_idx = np.maximum(0, np.min(indices, axis=0))
            max_idx = np.minimum(np.array(size) - 1, np.max(indices, axis=0))

            if np.any(min_idx > max_idx):
                continue

            x_range = np.arange(min_idx[0], max_idx[0] + 1)
            y_range = np.arange(min_idx[1], max_idx[1] + 1)
            z_range = np.arange(min_idx[2], max_idx[2] + 1)

            xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')

            indices_flat = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
            physical_points = indices_flat @ affine.T + origin_arr

            diff = physical_points - center

            if sphere:
                dist_sq = np.sum(diff ** 2, axis=1)
                inside = dist_sq <= current_radius ** 2

            elif cylinder:
                mag = np.linalg.norm(diff, axis=1)
                orientation_mag = np.linalg.norm(orientation)

                if orientation_mag == 0:
                    inside = np.zeros_like(mag, dtype=bool)
                else:
                    orientation = orientation / orientation_mag
                    proj = np.dot(diff, orientation)

                    proj_abs = np.abs(proj)
                    dist_to_axis = np.sqrt(np.clip(mag ** 2 - proj ** 2, 0, None))

                    inside = (proj_abs <= height / 2.0) & (dist_to_axis <= current_radius)

            inside = inside.reshape(xx.shape)

            out_lm[min_idx[0]:max_idx[0] + 1,
                   min_idx[1]:max_idx[1] + 1,
                   min_idx[2]:max_idx[2] + 1][inside] = foregroundLabel

        print("Writing label map...")
        self.io.write_from_numpy(out_lm, metainfo, output_path)
        print("DONE.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip',
                        required=True,
                        help="Input particles file name")
    parser.add_argument('--ilm',
                        required=True,
                        help="Input label map file name. Used to retrieve spacing, origin, and dimensions for "
                             "creating output label map")
    parser.add_argument('--o',
                        required=True,
                        help="Output label map file name")
    parser.add_argument('--cylinder',
                        action='store_true',
                        help="Set this flag to indicate that the cylinder stencil should be used")
    parser.add_argument('--sphere',
                        action='store_true',
                        help="Set this flag to indicate that the sphere stencil should be used.")
    parser.add_argument('--radius',
                        type=float,
                        default=1.0,
                        help="Stencil radius in mm. (default: 1.0)")
    parser.add_argument('--height',
                        type=float,
                        default=1.0,
                        help="Cylinder stencil height in mm. Default is 1mm. This should typically be set to "
                             "the inter-particle distance. (default: 1)")
    parser.add_argument('--sigma',
                        type=float,
                        default=0.0,
                        help="The CT scanner point spread function sigma. 0.0 by default. (default: 0)")
    parser.add_argument('--scale',
                        action='store_true',
                        help="Setting this flag will cause the stencil pattern to be scaled according to particle "
                             "scale. If set, any radius value specified using the --radius flag will be ignored. "
                             "Scaling will be performed using predetermined equations relating particle scale and "
                             "CT point spread function sigma (set using the --sigma flag)")
    parser.add_argument('--dnn_radius',
                        action='store_true',
                        help="Setting this flag will cause the stencil pattern to be scaled according to particle "
                             "dnn radius. If set, any radius value specified using the --radius flag will be ignored.")
    parser.add_argument('--dnn_radius_name',
                        type=str,
                        default="",
                        help="Name of the particles array containing dnn radius information. If not specified, "
                             "no information will be used for generating the stencil.")
    parser.add_argument('--airway',
                        action='store_true',
                        help="Set this flag to indicate that in the input particles correspond to airways.")
    parser.add_argument('--vessel',
                        action='store_true',
                        help="Set this flag to indicate that in the input particles correspond to vessels.")
    parser.add_argument('--fissure',
                        action='store_true',
                        help="Set this flag to indicate that in the input particles correspond to fissures.")
    args = parser.parse_args()

    slm = StenciledLabelMap()
    slm.execute(args.ip, args.ilm, args.o, args.airway, args.vessel, args.fissure, args.sigma, args.radius,
                args.height, args.cylinder, args.sphere, args.scale, args.dnn_radius, args.dnn_radius_name)