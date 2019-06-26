import argparse
import SimpleITK as sitk

from cip_python.common import ChestConventions

class ExampleCLIClass(object):
    def execute(self, input_file_name, output_file_name, variance=25.0, max_kernel_width=32):
        """
        Execute an ITK Gaussian filter
        Args:
            input_file_name: str. Path to the input volume/image
            output_file_name: str. Path where the ouput will be stored
            variance: float. Filter variance
            max_kernel_width: int. Maximum filter kernel width

        Returns:
            SimpleITK result volume
        """
        # Read the image
        input = sitk.ReadImage(input_file_name)
        # Apply filter
        filter = sitk.DiscreteGaussianImageFilter()
        filter.SetVariance(variance)
        filter.SetMaximumKernelWidth(max_kernel_width)
        output = filter.Execute(input)
        # Write result
        sitk.WriteImage(output, output_file_name)
        print("File saved to: {}".format(output_file_name))
        return output


if __name__ == "__main__":
    desc = "This is just an example that may be used as a template to develop new python CLIs"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("-i", "--input", dest="inputFileName", type=str, help="Input file name", required=True)
    parser.add_argument("-o", "--output", dest="outputFileName", type=str, help="Output file name", required=True)
    parser.add_argument("-v", "--variance", dest="gaussianVariance", type=float, help="Variance to use in the smooth filter", default=25.0)
    parser.add_argument("-k", "--maxKernelWidth", type=int, help="Maximum size of the kernel filter used in the smooth filter", default=32)

    args=parser.parse_args()

    print("These are the current Chest Regions available: ")
    for region in ChestConventions.ChestRegionsCollection.keys():
        print(ChestConventions.GetChestRegionName(region))

    cli = ExampleCLIClass()
    cli.execute(args.inputFileName, args.outputFileName, args.gaussianVariance, args.maxKernelWidth)


