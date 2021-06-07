import SimpleITK as sitk
import argparse


def resample_image_itk_by_reference(labelmap, img_ref, interpolator=sitk.sitkNearestNeighbor):

        reader = sitk.ImageFileReader()
        reader.SetImageIO("NrrdImageIO")
        reader.SetFileName(labelmap)
        lm = reader.Execute();
        reader.SetFileName(img_ref)
        ref_nrrd = reader.Execute();
    

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(interpolator)
        resampler.SetOutputDirection(ref_nrrd.GetDirection())
        resampler.SetOutputOrigin(ref_nrrd.GetOrigin())
        resampler.SetSize(ref_nrrd.GetSize())
        resampler.SetOutputSpacing(ref_nrrd.GetSpacing())
    
        return resampler.Execute(lm)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Resample labelmap to reference nrrd')
    parser.add_argument("-in_lm", dest="in_lm", required=True)
    parser.add_argument("-out_lm", dest="out_lm", required=True)
    parser.add_argument("-ref", dest="img_ref", required=True)

op = parser.parse_args()

resampled_lm = resample_image_itk_by_reference(op.in_lm, op.img_ref)

writer = sitk.ImageFileWriter()
writer.SetImageIO("NrrdImageIO")
writer.SetFileName(op.out_lm)
writer.UseCompressionOn()
writer.Execute(resampled_lm)

