from optparse import OptionParser
from cip_python.input_output import ImageReaderWriter
from cip_python.quality_control.qc_base import ImageOverlay
from cip_python.quality_control.qc_base import Montage
import numpy as np

class CTRegistrationQC:
    """
    General purpose class that generates CT QC images.
    """
    
    def __init__(self):
        pass
        
            
    def execute(self, in_ct1, in_ct2, out_file, in_mask, num_images, 
        overlay_alpha=0.5, window_width=1100, window_level=-1024, resolution=600,
        spacing=None,is_axial=True, is_sagittal=False, is_coronal=False):
        
        if (is_axial==False and is_sagittal==False and is_coronal==False):
            raise ValueError("No axes specified.")
        my_overlay = ImageOverlay()
        list_cts = []
        list_of_voxel_spacing=[]
        if(is_axial):
            list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images, axis='axial')) 
            list_of_voxel_spacing.append([[spacing[1],spacing[0]]]*num_images)       
            # Append the checkerboard image
            list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images,axis='axial', is_checkerboard=True))
            list_of_voxel_spacing.append([[spacing[1],spacing[0]]]*num_images)  
        if(is_sagittal):
            #list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images, axis='sagittal'))        
            list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images, axis='sagittal'))   
            list_of_voxel_spacing.append([[spacing[2],spacing[1]]]*num_images)     
            # Append the checkerboard image
            list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images,axis='sagittal', is_checkerboard=True))
            list_of_voxel_spacing.append([[spacing[2],spacing[1]]]*num_images) 
        if(is_coronal):
            list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images, axis='coronal'))        
            list_of_voxel_spacing.append([[spacing[2],spacing[0]]]*num_images)
            # Append the checkerboard image
            list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images,axis='coronal', is_checkerboard=True))
            list_of_voxel_spacing.append([[spacing[2],spacing[0]]]*num_images) 
                      
        my_montage = Montage() 
        my_montage.execute(list_cts,[], out_file, overlay_alpha, len(list_cts), num_images, \
            window_width=window_width, window_level=window_level, resolution=resolution, \
            list_of_voxel_spacing=list_of_voxel_spacing)  

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
                      metavar='<string>', type=int, default=3)   
    parser.add_option('--window_width',
                      help='intensity window width .  (optional)',  dest='window_width', 
                      metavar='<string>', type=int, default=1100)                         
                                           
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', type=int, default=-1024)                                                                 
    parser.add_option('--output_file',
                      help='QC image output file name.', dest='output_file', metavar='<string>',
                      default=None)   
    parser.add_option('--overlay_opacity',
                    help='Opacity of QC overlay or blend between ct1 and ct2 (between 0 and 1)',  dest='overlay_opacity', 
                    metavar='<string>', type=float, default=0.5)   	                                                                            
    parser.add_option('--resolution',
                      help='Output image resolution (dpi).  (optional)',  dest='output_image_resolution', 
                      metavar='<string>', type=float,  default=600)   
    parser.add_option("--axial", 
                      help='Set if want to QC CT images in the axial plane', dest="axial", 
                      action='store_true')   	                                                                            	                                                                            	                                                                            
    parser.add_option("--sagittal", 
                      help='Set if want to QC CT images in the sagittal plane', dest="sagittal", 
                      action='store_true')  
    parser.add_option("--coronal", 
                      help='Set if want to QC CT images in the coronal plane', dest="coronal", 
                      action='store_true')  
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    print "Reading CT..."
    ct_array_fixed, ct_header = image_io.read_in_numpy(options.in_ct_fixed) 
    ct_array_moving, ct_header = image_io.read_in_numpy(options.in_ct_moving) 
    partial_array=np.ones_like(ct_array_fixed)
    if(options.in_partial):
        partial_array, partial_header = image_io.read_in_numpy(options.in_partial) 

    spacing=ct_header['spacing']

    my_ct_qc = CTRegistrationQC()        
    my_ct_qc.execute(ct_array_fixed, ct_array_moving, options.output_file, partial_array, options.num_images_per_region, \
        overlay_alpha=options.overlay_opacity, window_width=options.window_width, window_level=options.window_level, \
        resolution=options.output_image_resolution, spacing=spacing,is_axial=options.axial, is_sagittal=options.sagittal, \
        is_coronal=options.coronal)
