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

    parser = argparse.ArgumentParser(description='Resample image to the coordinate frame of a reference image')
    parser.add_argument("-i", dest="in_im", required=True)
    parser.add_argument("-o", dest="out_im", required=True)
    parser.add_argument("--ref", dest="im_ref", required=True)
    parser.add_argument("--interpolator", dest="interpolator", choices=['NearestNeighbor','Linear','BSpline','BSpline1','BSpline2','BSpline3','BSpline4','BSpline5',
        'WelchWindowedSinc','HammingWindowedSinc','CosineWindowedSinc','BlackmanWindowedSinc','LanczosWindowSinc'],default='Linear')

op = parser.parse_args()

interpolator=dict()
interpolator['NearestNeighbor']=sitk.sitkNearestNeighbor
interpolator['Linear']=sitk.sitkLinear
interpolator['BSpline']=sitk.sitkBSpline
interpolator['BSpline1']=sitk.sitkBSpline1
interpolator['BSpline2']=sitk.sitkBSpline2
interpolator['BSpline3']=sitk.sitkBSpline3
interpolator['BSpline4']=sitk.sitkBSpline4
interpolator['BSpline5']=sitk.sitkBSpline5
interpolator['WelchWindowedSinc']=sitk.sitkWelchWindowedSinc
interpolator['HammingWindowedSinc']=sitk.sitkHammingWindowedSinc

resampled_im = resample_image_itk_by_reference(op.in_im, op.im_ref,interpolator=interpolator[op.interpolator])

writer = sitk.ImageFileWriter()
writer.SetImageIO("NrrdImageIO")
writer.SetFileName(op.out_im)
writer.UseCompressionOn()
writer.Execute(resampled_im)

