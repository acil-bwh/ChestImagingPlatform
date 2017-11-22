import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import nrrd


class Extract2DPatchesFromParticles:
    def __init__(self, patch_size, output_spacing=[0.5, 0.5, 0.5], structure_type='airway'):
        self._patch_size = patch_size
        self._output_spacing = output_spacing
        self._structure_type = structure_type

    def execute(self, ct_image, pp_file, output_filepath):
        vtk_reader = vtk.vtkNrrdReader()
        vtk_reader.SetFileName(ct_image)
        vtk_reader.Update()
        ct_image = vtk_reader.GetOutput()

        input_spacing = np.asarray(ct_image.GetSpacing())
        geom_mean = np.prod(input_spacing) ** (1 / 3.0)
        self._output_spacing[2] = geom_mean

        print self._output_spacing

        factor = np.asarray(ct_image.GetSpacing()) / np.asarray(self._output_spacing)

        res = vtk.vtkImageResample()
        res.SetOutputSpacing(self._output_spacing)
        res.SetAxisMagnificationFactor(0, factor[0])
        res.SetAxisMagnificationFactor(1, factor[1])
        res.SetAxisMagnificationFactor(2, factor[2])
        res.SetInputData(ct_image)
        res.Update()
        resampled_ct_image = res.GetOutput()

        rr = vtk.vtkPolyDataReader()
        rr.SetFileName(pp_file)
        rr.Update()
        pp = rr.GetOutput()

        patches_ = list()

        for point in range(pp.GetNumberOfPoints()):
            center = pp.GetPoint(point)

            hevec0 = pp.GetPointData().GetArray('hevec0').GetTuple3(point)
            hevec1 = pp.GetPointData().GetArray('hevec1').GetTuple3(point)
            hevec2 = pp.GetPointData().GetArray('hevec2').GetTuple3(point)

            if self._structure_type == 'vessel':
                XAxis = hevec1
                YAxis = hevec2
                ZAxis = hevec0
            elif self._structure_type == 'airway':
                XAxis = hevec0
                YAxis = hevec1
                ZAxis = hevec2
            else:
                raise Exception("airway and vessel are the only types allowed. {} specified".format(op.type))

            img = self.reslice_image_2D(resampled_ct_image, XAxis, YAxis, ZAxis, center, self._patch_size)

            img_patch = vtk_to_numpy(img)
            img_patch = img_patch.reshape(self._patch_size[0], self._patch_size[1], 1)
            img_patch = img_patch.squeeze()
            patches_.append(img_patch)

        patches_file = np.transpose(patches_, [2, 1, 0])

        nrrd_dict = dict()
        nrrd_dict['spacings'] = self._output_spacing
        nrrd.write(output_filepath, patches_file, options=nrrd_dict)

    def reslice_image_2D(self, image, x, y, z, center, size):
        reslice = vtk.vtkImageReslice()
        reslice.SetInputData(image)
        reslice.SetResliceAxesDirectionCosines(x[0], x[1], x[2], y[0], y[1], y[2], z[0],
                                               z[1], z[2])
        reslice.SetResliceAxesOrigin(center)
        reslice.SetOutputDimensionality(2)

        reslice.SetInterpolationMode(vtk.VTK_RESLICE_CUBIC)
        reslice.SetOutputSpacing(self._output_spacing)
        reslice.SetOutputExtent(0, size[0] - 1, 0, size[1] - 1, 0, 1)
        reslice.SetOutputOrigin(-(size[0] * 0.5 - 0.5) * self._output_spacing[0],
                                -(size[1] * 0.5 - 0.5) * self._output_spacing[1],
                                0)
        # reslice.SetNumberOfThreads(4)
        reslice.Update()
        return reslice.GetOutput().GetPointData().GetScalars()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Method to get 2D patches from particle points with reslicing')
    parser.add_argument('--ict', dest='in_ct', help="Input CT image", type=str, required=True)
    parser.add_argument('--ipp', dest='in_pp', help="Input VTK particle file", type=str, required=True)
    parser.add_argument('--sz', dest='patch_sz', help="2D size of the output patches. Size should be specified as "
                                                      "x,y. Along z, the mean inter particle distance is used",
                        type=str, required=True)
    parser.add_argument('--sp', dest='spacing', help="Output spacing. Spacing should be specified as x,y. Along z, "
                                                     "the mean inter particle distance is used",
                        type=str, required=True)
    parser.add_argument('--type', dest='type', help="Type of patches for reslicing. Current allowed types are: airway, "
                                                    "vessel",
                        type=str, required=True)
    parser.add_argument('--o', dest='output_file', help="Output NRRD file", type=str, required=True)

    op = parser.parse_args()

    patch_sz = [int(sz) for sz in op.patch_sz.split(',')]
    out_spacing = [float(sp) for sp in op.spacing.split(',')]
    out_spacing.append(1.0)  # Necessary for resampling

    ct_image = op.in_ct
    pp_file = op.in_pp
    output_file = op.output_file

    ep = Extract2DPatchesFromParticles(patch_sz, output_spacing=out_spacing, structure_type=op.type)
    ep.execute(op.in_ct, op.in_pp, op.output_file)



