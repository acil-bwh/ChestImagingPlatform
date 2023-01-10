"""
File: tx2dfield.py
Author: Ariel Hernán Curiale
Email: curiale@gmail.com
Github: https://gitlab.com/Curiale
Description:
    This class return the dense deformation field from a list of
    transformations
"""
import SimpleITK as sitk
from cip_python.registration.transform import CompositeTransform


class Transform2DenseField():
    """
    Filter to create a dense deformation field from a list of ITK Transform
    """

    def __init__(self, tx, ref_img):
        if not isinstance(tx, list) and not isinstance(tx, tuple):
            tx = [tx]

        if len(tx) == 1:
            self.tx = tx[0]
        else:
            self.tx = CompositeTransform(tx)
        self.ref = ref_img

    def Execute(self):
        """
        Returns the dense Displacement Field
        """
        dfield_filter = sitk.TransformToDisplacementFieldFilter()
        dfield_filter.SetReferenceImage(self.ref)
        dfield = dfield_filter.Execute(self.tx)
        return dfield


def main():
    import argparse
    usage = "usage: given a image reference and a lis of ITK transformations, "
    usage += " create the dense deformation field.\n"
    usage += "NOTE: First Tx passed is the last Tx applied, for emxaple, the"
    usage += " orden should be [Affine, FFD]"
    parser = argparse.ArgumentParser(description=usage)
    parser.add_argument('-r',
                        '--ref',
                        help="Reference image",
                        type=str,
                        required=True)
    parser.add_argument('-s',
                        '--save',
                        help="Save file",
                        type=str,
                        required=True)
    parser.add_argument(
        '-t',
        '--transforms',
        nargs='+',
        help="List of transformations, for example [Affine, FFD]",
        type=list,
        required=True)

    args = parser.parse_args()
    # Read image
    ref_img = sitk.ReadImage(args.ref)
    # Read ITK Transfomrations
    tx = [sitk.ReadTransform(tx_name) for tx_name in args.transforms]
    tx2dfield = Transform2DenseField(tx, ref_img)
    dfield = tx2dfield.Execute()
    sitk.WriteImage(dfield, args.save, True)


if __name__ == "__main__":
    main()
