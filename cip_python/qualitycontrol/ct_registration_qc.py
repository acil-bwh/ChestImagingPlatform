from optparse import OptionParser
from cip_python.input_output import ImageReaderWriter
from  cip_python.qualitycontrol.qc_base import ImageOverlay
from  cip_python.qualitycontrol.qc_base import Montage

class CTRegistrationQC:
    """
    General purpose class that generates CT QC images.
    """
    
    def __init__(self):
        pass
        
            
    def execute(self, in_ct1, in_ct2, out_file, in_mask, num_images, 
        overlay_alpha=0.5, window_width=1100, window_level=-1024, resolution=600):
        
        my_overlay = ImageOverlay()
        list_cts = []
        list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images))
                
        # Append the checkerboard image
        list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images,is_checkerboard=True))
        
        my_montage = Montage() 
        my_montage.execute(list_cts,[], out_file, overlay_alpha, len(list_cts), num_images, \
            window_width=window_width, window_level=window_level, resolution=resolution)  

if __name__ == "__main__":
    desc = """Generates a montage of slices extracted from 2 registered CT volumes for QC purposes."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct_fixed',
                      help='Input ct file name. ', dest='in_ct_fixed', metavar='<string>',
                      default=None)          
    parser.add_option('--in_ct_moving',
                      help='Input moving ct file name. ', dest='in_ct_moving', metavar='<string>',
                      default=None)  
    parser.add_option('--in_partial',
                      help='Input lung labelmap file name. Used as a mask \
                      to set the Qc bounding box ', dest='in_partial', metavar='<string>',
                      default=None)
    parser.add_option('--num_images_per_region',
                      help='Number of images to be output per region requested to QC.Images will be extracted \
                      evenly throughout the region. This does not apply to CT QC, where 1 image per axis is output (optional)',  dest='num_images_per_region', 
                      metavar='<string>', default=3)   
    parser.add_option('--window_width',
                      help='intensity window width .  (optional)',  dest='window_width', 
                      metavar='<string>', default=1100)                         
                                           
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', default=-1024)                                                                 
    parser.add_option('--output_file',
                      help='QC image output file name.', dest='output_file', metavar='<string>',
                      default=None)   
    parser.add_option('--overlay_opacity',
                    help='Opacity of QC overlay or blend between ct1 and ct2 (between 0 and 1)',  dest='overlay_opacity', 
                    metavar='<string>', type=float, default=0.5)   	                                                                            
    parser.add_option('--resolution',
                      help='Output image resolution (dpi).  (optional)',  dest='output_image_resolution', 
                      metavar='<string>', type=float,  default=600)   	                                                                            	                                                                            	                                                                            
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    print "Reading CT..."
    ct_array_fixed, ct_header = image_io.read_in_numpy(options.in_ct_fixed) 
    ct_array_moving, ct_header = image_io.read_in_numpy(options.in_ct_moving) 
    partial_array, partial_header = image_io.read_in_numpy(options.in_partial) 
    
    my_ct_qc = CTRegistrationQC()        
    my_ct_qc.execute(ct_array_fixed, ct_array_moving, options.output_file, partial_array, options.num_images_per_region, \
        overlay_alpha=options.overlay_opacity, window_width=options.window_width, window_level=options.window_level, \
        resolution=options.output_image_resolution)