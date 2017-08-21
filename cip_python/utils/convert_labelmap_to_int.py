# Convert a labelmap from float to unsigned int
import SimpleITK as sitk
import sys

def convert_labelmap_to_int(labelmap_path, output_path):
    original = sitk.ReadImage(labelmap_path)
    filter = sitk.CastImageFilter()
    filter.SetOutputPixelType(3)    # 16-bit unsigned integer
    convert = filter.Execute(original)
    # Write result (last param: useCompression)
    sitk.WriteImage(convert, output_path, True)
    print ("{} written".format(output_path))

if __name__ == "__main__":
    convert_labelmap_to_int(sys.argv[1], sys.argv[2])
