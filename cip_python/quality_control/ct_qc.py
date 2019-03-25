from optparse import OptionParser
from cip_python.input_output import ImageReaderWriter
from cip_python.quality_control.qc_base import ImageOverlay
from cip_python.quality_control.qc_base import Montage

class CTQC:
    """
    General purpose class that generates CT QC images.
    """
    def __init__(self):
        pass
            
    def execute(self, in_ct, out_file, window_width=1100, window_level=-1024, resolution=50,
        spacing=None):

        my_projection = ImageOverlay() 
        
        x_projection = my_projection.get_ct_projection(in_ct, axis='sagittal')
        y_projection = my_projection.get_ct_projection(in_ct, axis='coronal')
        z_projection = my_projection.get_ct_projection(in_ct, axis='axial')
        list_of_voxel_spacing=[]
        list_of_voxel_spacing.append([spacing[2],spacing[1]])
        list_of_voxel_spacing.append([spacing[2],spacing[0]])
        list_of_voxel_spacing.append([spacing[1],spacing[0]])

        # combine into 1 image, note: the images have already been rotated by 90 degrees for 
        # display purposes
        my_montage = Montage()
        import numpy as np

        my_montage.execute([[[x_projection], [y_projection], [z_projection]]],[], \
            out_file, 0.0, 1, 3, window_width=window_width, window_level=window_level,\
            resolution=resolution, list_of_voxel_spacing=[list_of_voxel_spacing])                        
            
if __name__ == "__main__":
    desc = """Generates a montage of CT slices from a nrrd volume for QC purposes."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input ct file name for which QC is desired. ', dest='in_ct', metavar='<string>',
                      default=None)            
    parser.add_option('--window_width',
                      help='intensity window width .  (optional)',  dest='window_width', 
                      metavar='<string>', default=1100)                                                                    
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', default=-1024)                                                                 
    parser.add_option('--output_file',
                      help='QC image output file name.', dest='output_file', metavar='<string>',
                      default=None)                                                                               
    parser.add_option('--resolution',
                      help='Output image resolution (dpi).  (optional)',  dest='output_image_resolution', 
                      metavar='<string>', type=float,  default=600)   	                                                                            	                                                                            	                                                                            
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    print ("Reading CT...")
    ct_array, ct_header = image_io.read_in_numpy(options.in_ct) 
    spacing=ct_header['spacing']
    my_ct_qc = CTQC()
    my_ct_qc.execute(ct_array, options.output_file,  \
        window_width=options.window_width, window_level=options.window_level, \
        resolution=options.output_image_resolution, spacing=spacing)
    

               

