import numpy as np
from optparse import OptionParser
import warnings
from cip_python.input_output import ImageReaderWriter
from  cip_python.qualitycontrol.qc_base import LabelmapQC

class LungSegmentationQC(LabelmapQC):
    """
    General purpose class that generates lung segmentation QC images.
    
    Parameters 
    ---------- 
      
    Npne
        
    """
    
    def __init__(self):

        self.region_dictionary  = {\
        'leftLung': ['LEFTUPPERTHIRD','LEFTMIDDLETHIRD','LEFTLOWERTHIRD'],\
        'rightLung': ['RIGHTUPPERTHIRD','RIGHTMIDDLETHIRD','RIGHTLOWERTHIRD'],\
        'leftLungLobes': ['LEFTSUPERIORLOBE','LEFTINFERIORLOBE'],\
        'rightLungLobes' : ['RIGHTSUPERIORLOBE', 'RIGHTMIDDLELOBE',\
                'RIGHTINFERIORLOBE'] 
                }   
                
        self.labelmap_qc_dictionary={'partialLungLabelmap':['leftLung','rightLung'],\
            'lungLobeLabelmap':['leftLungLobes','rightLungLobes']}
        
     
        LabelmapQC.__init__(self)  
                             
    def execute(self, in_ct, out_file, qc_requested, num_images_per_region=3, in_partial=None, in_lobe=None, \
            overlay_alpha=0.85, window_width=1100, window_level=-1024, resolution=600, spacing=None):   
        
        list_of_labelmaps=[] 
        list_request_qc_per_labelmap=[]
        
        """ add the labelmaps needed"""
        for labelmaptype in self.labelmap_qc_dictionary.keys():
            labelmap_qc_requested = set(self.labelmap_qc_dictionary[labelmaptype]).intersection(set(qc_requested)) 
            if len(labelmap_qc_requested)>0:
                """ the labelmap tye is needed for some requested QC, thus add it to list"""
                if (labelmaptype=='partialLungLabelmap'):
                    list_of_labelmaps.append(in_partial)
                if (labelmaptype=='lungLobeLabelmap'):
                    list_of_labelmaps.append(in_lobe)
                   
                list_request_qc_per_labelmap.append(list(labelmap_qc_requested))       
                
        
        self.get_labelmap_qc(in_ct, out_file, list_of_labelmaps, list_request_qc_per_labelmap, \
            num_images_per_region=num_images_per_region, \
            overlay_alpha=overlay_alpha, window_width=window_width, window_level=window_level,\
            axis='sagittal', resolution=resolution,spacing=spacing )



if __name__ == "__main__":
    desc = """Generates a montage of slices extracted from volumes for QC purposes."""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input ct file name. ', dest='in_ct', metavar='<string>',
                      default=None)          
    parser.add_option('--in_partial',
                      help='Input lung labelmap file name. If input \
                      lung labelmaps will be QCed, ', dest='in_partial', metavar='<string>',
                      default=None)
    parser.add_option('--in_lobes',
                      help='Input lung lobes file name', dest='in_lobes', metavar='<string>',
                      default=None)                                                        
    parser.add_option('--num_images_per_region',
                      help='Number of images to be output per region requested to QC.Images will be extracted \
                      evenly throughout the region. This does not apply to CT QC, where 1 image per axis is output (optional)',  dest='num_images_per_region', 
                      metavar='<string>', type=int, default=3)   
    parser.add_option('--window_width',
                      help='intensity window width .  (optional)',  dest='window_width', 
                      metavar='<string>',type=int, default=1100)                                                                    
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', type=int,default=-1024)                                                                 
    parser.add_option("--qc_leftlung", 
                      help='Set to true if want to QC left partial lung labelmap images', 
                      dest="qc_leftlung", action='store_true')  
    parser.add_option("--qc_rightlung", 
                      help='Set to true if want to QC right partial lung labelmap images', 
                      dest="qc_rightlung", action='store_true')  
    parser.add_option("--qc_leftlunglobes", 
                      help='Set to true if want to QC left lung lobe images', 
                      dest="qc_leftlunglobes", action='store_true')  
    parser.add_option("--qc_rightlunglobes", 
                      help='Set to true if want to QC right lung lobe images', 
                      dest="qc_rightlunglobes", action='store_true')  
    parser.add_option('--output_file',
                      help='QC image output file name.', dest='output_file', metavar='<string>',
                      default=None)   
    parser.add_option('--overlay_opacity',
                    help='Opacity of QC overlay (between 0 and 1)',  dest='overlay_opacity', 
                    metavar='<string>', type=float, default=0.85)   	                                                                            
    parser.add_option('--resolution',
                      help='Output image resolution (dpi).  (optional)',  dest='output_image_resolution', 
                      metavar='<string>', type=float,  default=600)   	                                                                            	                                                                            	                                                                            
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    print "Reading CT..."
    ct_array, ct_header = image_io.read_in_numpy(options.in_ct) 
    
    partial_array = None
    print "Reading partial lung labelmap..."
    if(options.in_partial):
        partial_array, partial_header = image_io.read_in_numpy(options.in_partial) 
    
    lobe_array = None
    if(options.in_lobes):
        print "Reading lobe..."
        lobe_array, lobe_header = image_io.read_in_numpy(options.in_lobes)               
        
    list_of_lung_qc = []
    if(options.qc_leftlung):
       list_of_lung_qc.append('leftLung') 
    if(options.qc_rightlung):
       list_of_lung_qc.append('rightLung') 
    if(options.qc_leftlunglobes):
       list_of_lung_qc.append('leftLungLobes') 
    if(options.qc_rightlunglobes):
       list_of_lung_qc.append('rightLungLobes') 

    if(len(list_of_lung_qc) > 0):              
        print("generating lung qc for: ")
        print(list_of_lung_qc)
        spacing=ct_header['spacing']
        my_lung_qc = LungSegmentationQC()
        my_lung_qc.execute(ct_array, options.output_file, qc_requested=list_of_lung_qc, \
            num_images_per_region=options.num_images_per_region, in_partial=partial_array, \
            in_lobe=lobe_array, overlay_alpha=options.overlay_opacity, window_width=options.window_width, \
            window_level=options.window_level, resolution=options.output_image_resolution, spacing=spacing)

