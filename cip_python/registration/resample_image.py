
import SimpleITK as sitk
import re
from cip_python.registration.transform import CompositeTransform
from cip_python.registration.tx2dfield import Transform2DenseField


def cast(img, pixelID):
    """
    Useful function for casting ITK images
    """
    if img.GetPixelID() == pixelID:
        return img

    if ((pixelID == sitk.sitkUInt8) or (pixelID == sitk.sitkUInt16)
            or (pixelID == sitk.sitkUInt32) or (pixelID == sitk.sitkUInt64)
            or (pixelID == sitk.sitkInt8) or (pixelID == sitk.sitkInt16)
            or (pixelID == sitk.sitkInt32) or (pixelID == sitk.sitkInt64)):
        img = sitk.Round(img)
    img = sitk.Cast(img, pixelID)
    return img


class ResampleImage():
    """
    This class resamples the image mov into the Fix coord. system using an
    inverse mapping provided by the transformation:
    1) Inverse mapping mean that pout = pin + delta
    2) the resampling is done by applying the inverse mapping as follows:
        I_warp(pout) = I_mov(pin + delta)
    where I_warp is in the F coord. system and I_mov is in M the coord. system.

    So, transform.TransformPoint move point from the Fix coord. system to
    the Moving coord. system.

    The resampling of the moving image (I_M --> I_F) is done without taking
    care about the change in the volumes. To compute volumes or densities after
    resampling use resample_with_jacobian instead for properly deal with the
    volume compressino/expansion.

    """

    def __init__(self, fix, mov, tx):
        if not isinstance(tx, list) and not isinstance(tx, tuple):
            tx = [tx]

        if len(tx) == 1:
            self.tx = tx[0]
        else:
            self.tx = CompositeTransform(tx)
        self.fix = fix
        self.mov = mov

    def Execute(self,
                intensity_correction=False,
                offset_correction=0,
                interpolator=None,
                is_label=False,
                default_value=0):
        """
        Moving image is resample match the Fix image accordint to the Txs.
        i.e., Moving --> Fix and warped image is defined in the Fix coord.
        system sapce.
        """

        # Resample Moving --> Fixed
        # This variable is used to explicity define that the fix image is the
        # reference image
        ref_img = self.fix
        if interpolator is None:
            interpolator = sitk.sitkLinear
        if is_label:
            interpolator = sitk.sitkNearestNeighbor

        img_warpped = sitk.Resample(self.mov, ref_img, self.tx, interpolator,
                                    default_value)

        if intensity_correction:
            pixelID = img_warpped.GetPixelID()
            use_img_spacing = True
            # Get the Dense field from a Tx
            tx2dfield = Transform2DenseField(self.tx, ref_img)
            dfield = tx2dfield.Execute()
            J = sitk.DisplacementFieldJacobianDeterminant(
                dfield, use_img_spacing)
            # Compensate volume/density
            # Cast to Float64 to multiply with J which is Float64
            img_warpped = sitk.Cast(img_warpped, sitk.sitkFloat64)
            # Shift according to the offset before compensation
            img_warpped = (img_warpped + offset_correction) * J
            img_warpped = img_warpped - offset_correction
            # Cast to the original pixel type
            img_warpped = cast(img_warpped, pixelID)

        return img_warpped


def main():
    import argparse
    usage = (
        "usage: given fixed and moving images, and a lis of "
        "ITK transformations, this script resample the moving image into the "
        "coordinate space defined by the reference image.\n"
        "NOTE: First Tx passed is the last Tx applied, for emxaple, the"
        " orden should be [Affine, FFD]")
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('-r',
                        '--ref',
                        help="Reference image",
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--moving',
                        help="Moving image",
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        help="Resampled moving image file",
                        type=str,
                        required=True)
    parser.add_argument(
        '-t',
        '--transforms',
        nargs='+',
        help="List of transformations, for example [Affine, FFD]",
        type=str,
        required=True)
    parser.add_argument("--interpolator", dest="interpolator", choices=['NearestNeighbor','Linear','BSpline','BSpline1','BSpline2','BSpline3','BSpline4','BSpline5',
        'WelchWindowedSinc','HammingWindowedSinc','CosineWindowedSinc','BlackmanWindowedSinc','LanczosWindowSinc'],default='Linear')
    parser.add_argument('-c', '--correction', action='store_true')
    parser.add_argument('-oc', '--offset_correction', type=float, default=0)

    args = parser.parse_args()
    # Read images
    fix = sitk.ReadImage(args.ref)
    mov = sitk.ReadImage(args.moving)
    # Read ITK Transfomrations
    #Read transforms
    tx=list()
    T_filenames=args.transforms
    for tt in args.transforms:
        res = re.findall(r'\[(.*?)\]', tt)
        if len(res)==0:
            t_file=tt
        elif len(res)>=1:
            res=res[0].split(',')
            t_file=res[0]
        T_tmp=sitk.ReadTransform(t_file)

        if len(res)>1:
            if res[1]=='1':
                #Invert transform
                print("Inverting transform")
                T_tmp=T_tmp.GetInverse()
        tx.append(T_tmp)

    #tx = [sitk.ReadTransform(tx_name) for tx_name in args.transforms]

    #Interpolator map
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


    resample = ResampleImage(fix, mov, tx)
    warpped = resample.Execute(intensity_correction=args.correction,
                               offset_correction=args.offset_correction,interpolator=interpolator[args.interpolator])
    sitk.WriteImage(warpped, args.output, True)


if __name__ == "__main__":
    main()
