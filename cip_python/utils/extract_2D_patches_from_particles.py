import numpy as np
import vtk
import SimpleITK as sitk
from vtk.util.numpy_support import vtk_to_numpy,numpy_to_vtk
import nrrd
from vtk.util.vtkImageImportFromArray import  vtkImageImportFromArray

class Extract2DPatchesFromParticles:
    def __init__(self, patch_size, output_spacing=[0.5, 0.5,0.5], structure_type='airway'):
        self._patch_size = patch_size
        self._output_spacing = output_spacing
        self._structure_type = structure_type

    def execute(self, ct_image, pp, output_filepath):

        input_spacing = np.asarray(ct_image.GetSpacing())
        geom_mean = np.prod(input_spacing) ** (1 / 3.0)
        self._output_spacing[2] = geom_mean
        factor = np.asarray(ct_image.GetSpacing()) / np.asarray(self._output_spacing)

        res = vtk.vtkImageResample()
        res.SetOutputSpacing(self._output_spacing)
        res.SetAxisMagnificationFactor(0, factor[0])
        res.SetAxisMagnificationFactor(1, factor[1])
        res.SetAxisMagnificationFactor(2, factor[2])
        res.SetInputData(ct_image)
        res.Update()
        print "resampling"
        resampled_ct_image = res.GetOutput()

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

            #img = self.reslice_image_2D(resampled_ct_image, XAxis, YAxis, ZAxis, center, self._patch_size)
            img = self.reslice_image_2D(ct_image, XAxis, YAxis, ZAxis, center, self._patch_size)

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
        reslice.SetOutputScalarType(vtk.VTK_SHORT)
        reslice.SetOutputExtent(0, size[0] - 1, 0, size[1] - 1, 0, 1)
        reslice.SetOutputOrigin(-(size[0] * 0.5 - 0.5) * self._output_spacing[0],
                                -(size[1] * 0.5 - 0.5) * self._output_spacing[1],
                                0)
        # reslice.SetNumberOfThreads(4)
        reslice.Update()
        return reslice.GetOutput().GetPointData().GetScalars()

    def sitk2vtk2(self,img):

        i2 = sitk.GetArrayFromImage(img)
        ii = vtkImageImportFromArray()

        size     = list(img.GetSize())
        origin   = list(img.GetOrigin())
        spacing  = list(img.GetSpacing())
        extent=[0, size[0]-1, 0, size[1]-1, 0, size[2]-1]
        ii.SetDataExtent(extent)
        ii.SetDataSpacing(spacing)
        ii.SetDataOrigin(origin)
        i2=np.copy(i2)

        ii.SetArray(i2)

        ii.Update()
        im_vtk=ii.GetOutput()
        return im_vtk


    def sitk2vtk(self,img):

        size     = list(img.GetSize())
        origin   = list(img.GetOrigin())
        spacing  = list(img.GetSpacing())

        pixelmap=dict()
        pixelmap[sitk.sitkInt8]=vtk.VTK_CHAR
        pixelmap[sitk.sitkUInt8]=vtk.VTK_UNSIGNED_CHAR
        pixelmap[sitk.sitkInt16]=vtk.VTK_SHORT
        pixelmap[sitk.sitkUInt16]=vtk.VTK_UNSIGNED_SHORT
        pixelmap[sitk.sitkInt32]=vtk.VTK_INT
        pixelmap[sitk.sitkUInt32]=vtk.VTK_UNSIGNED_INT
        pixelmap[sitk.sitkFloat32]=vtk.VTK_FLOAT
        pixelmap[sitk.sitkFloat64]=vtk.VTK_DOUBLE

        sitktype = img.GetPixelID()
        vtktype  = pixelmap[sitktype]
        print size
        print origin
        print spacing
        ncomp    = img.GetNumberOfComponentsPerPixel()
        print ncomp

        # there doesn't seem to be a way to specify the image orientation in VTK

        # convert the SimpleITK image to a numpy array
        i2 = sitk.GetArrayFromImage(img)


        #i2=i2.transpose([2,1,0])
        #import pylab
        #i2 = reshape(i2, size)

        i2_string = i2.tostring()

        print len(i2_string)
        # send the numpy array to VTK with a vtkImageImport object
        dataImporter = vtk.vtkImageImport()
        dataImporter.SetDataScalarType(vtktype)

        dataImporter.SetNumberOfScalarComponents(ncomp)
        dataImporter.SetDataExtent (0, size[0]-1, 0, size[1]-1, 0, size[2]-1)
        dataImporter.SetWholeExtent(0, size[0]-1, 0, size[1]-1, 0, size[2]-1)

        dataImporter.SetDataOrigin(origin)
        dataImporter.SetDataSpacing(spacing)

        #dataImporter.CopyImportVoidPointer( i2_string, len(i2_string) )
        dataImporter.SetImportVoidPointer(i2_string)

        # VTK expects 3-dimensional parameters
        if len(size) == 2:
            size.append(1)

        if len(origin) == 2:
            origin.append(0.0)

        if len(spacing) == 2:
            spacing.append(spacing[0])

        # Set the new VTK image's parameters
        #


        print "Updating importer"
        dataImporter.Update()
        print "Done importer"

        vtk_image = dataImporter.GetOutput()
        return vtk_image


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

    ct_img_sitk=sitk.ReadImage(op.in_ct)
    vtk_reader = vtk.vtkNrrdReader()
    vtk_reader.SetFileName(op.in_ct)
    vtk_reader.Update()
    ct_image_vtk = vtk_reader.GetOutput()

    particle_reader = vtk.vtkPolyDataReader()
    particle_reader.SetFileName(op.in_pp)
    particle_reader.Update()
    particles_vtk = particle_reader.GetOutput()

    ep = Extract2DPatchesFromParticles(patch_sz, output_spacing=out_spacing, structure_type=op.type)

    #ct_image_vtk=ep.sitk2vtk2(ct_img_sitk)

    #print ct_image_vtk

    #print ct_image_vtk.GetPointData().GetArray('scalars').GetComponent(10,0)

    ep.execute(ct_image_vtk,particles_vtk, op.output_file)



