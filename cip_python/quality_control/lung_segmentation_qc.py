import numpy as np
from optparse import OptionParser
import warnings
from cip_python.input_output import ImageReaderWriter
from  cip_python.quality_control.qc_base import LabelmapQC
from  cip_python.quality_control.qc_base import Montage

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
        'rightLungLobes' : ['RIGHTSUPERIORLOBE', 'RIGHTMIDDLELOBE','RIGHTINFERIORLOBE'],
        'all_regions' : ['WildCard'],
        'all_types' : None,
        'airways' : None}   

        self.type_dictionary  = {\
        'leftLung': None,\
        'rightLung': None,\
        'leftLungLobes': None, 
        'rightLungLobes' : None, 
        'airways' : 'airways',
        'all_regions' : None ,
        'all_types' : ['WildCard']
                }    #['HORIZONTALFISSURE','OBLIQUEFISSURE'],\
                
        self.axes_dictionary = {\
        'leftLung': 'sagittal',\
        'rightLung': 'sagittal',\
        'leftLungLobes': 'sagittal',
        'rightLungLobes' : 'sagittal',
        'airways' : 'coronal',\
        'all_regions' : 'coronal',\
        'all_types' :'coronal'
        }            
                                                
        self.labelmap_qc_dictionary={'partialLungLabelmap':['leftLung','rightLung','all_regions','all_types'], #,'all_regions','all_types'
            'lungLobeLabelmap':['leftLungLobes','rightLungLobes','all_regions','all_types']}
        
        self.overlay_qc = {'leftLung','rightLung','leftLungLobes','rightLungLobes'}
        
        self.projection_qc = {'all_regions','all_types'}

        self.region_type_dictionary = {'leftLung': None, 'rightLung': None, 'leftLungLobes': None, \
                                       'rightLungLobes': None, 'airways': 'airways', 'all_regions': None, \
                                       'all_types': ['WildCard']}


        LabelmapQC.__init__(self)
                             
    def execute(self, in_ct, out_file, qc_requested, num_images_per_region=3, in_partial=None, in_lobe=None, \
            in_overlay_alpha=0.85, window_width=1100, window_level=-1024, resolution=600, spacing=None):   
        
        list_of_labelmaps_overlay=[] 
        list_request_qc_per_labelmap_overlay=[]
        list_axes_overlay=[]
        list_of_labelmaps_projection=[]
        list_request_qc_per_labelmap_projection=[]
        list_axes_projection=[]
        
        """ first check which labelmaps are available """
        available_labelmaps = []
        if in_partial is not None:
            available_labelmaps.append('partialLungLabelmap')
        if in_lobe is not None:
            available_labelmaps.append('lungLobeLabelmap')
                                                          
        """ add the labelmaps needed and perform labelmap QC"""                                                          
        for labelmaptype in available_labelmaps: #self.labelmap_qc_dictionary.keys():
            labelmap_qc_requested = set(self.labelmap_qc_dictionary[labelmaptype]).intersection(set(qc_requested)) 
            labelmap_qc_requested_overlay = labelmap_qc_requested.intersection(set(self.overlay_qc)) 
            labelmap_qc_requested_projection = labelmap_qc_requested.intersection(set(self.projection_qc)) 

            if len(labelmap_qc_requested_overlay)>0:
                """ the labelmap tye is needed for some requested QC, thus add it to list"""
                if (labelmaptype=='partialLungLabelmap'):
                    list_of_labelmaps_overlay.append(in_partial)
                if (labelmaptype=='lungLobeLabelmap'):
                    list_of_labelmaps_overlay.append(in_lobe)
                   
                list_request_qc_per_labelmap_overlay.append(list(labelmap_qc_requested_overlay))       
                axes_temp=[]
                for qc in labelmap_qc_requested_overlay:
                    axes_temp.append(self.axes_dictionary[qc])
                list_axes_overlay.append(axes_temp)

            if ((len(labelmap_qc_requested_projection)>0) and (len(labelmap_qc_requested_overlay)>0)):
                """ the labelmap tye is needed for some requested QC, thus add it to list"""
                if (labelmaptype=='partialLungLabelmap'):
                    list_of_labelmaps_projection.append(in_partial)
                if (labelmaptype=='lungLobeLabelmap'):
                    list_of_labelmaps_projection.append(in_lobe)
                   
                list_request_qc_per_labelmap_projection.append(list(labelmap_qc_requested_projection))       
                axes_temp=[]
                for qc in labelmap_qc_requested_projection:
                    axes_temp.append(self.axes_dictionary[qc])
                list_axes_projection.append(axes_temp)       

        [list_cts,list_labelmaps, list_of_voxel_spacing] = self.get_labelmap_qc(in_ct, out_file, list_of_labelmaps_overlay, \
            list_request_qc_per_labelmap_overlay, list_axes_overlay, num_images_per_region=num_images_per_region, \
            overlay_alpha=in_overlay_alpha, window_width=window_width, window_level=window_level,\
            resolution=resolution,spacing=spacing, is_overlay=True )
                  

        #""" get all the required projections in 1 shot"""  

        [list_cts2,list_labelmaps2, list_of_voxel_spacing2] = self.get_labelmap_qc(in_ct, out_file, list_of_labelmaps_projection, \
            list_request_qc_per_labelmap_projection, list_axes_projection, num_images_per_region=1, \
            overlay_alpha=1.0, window_width=window_width, window_level=window_level,\
            resolution=resolution,spacing=spacing, is_projection=True, is_overlay=False )             
        
        """ Now arrange the layout."""
        """ if QC with qc_regionstypes then append to the first 2 rows."""     
        """ append the 2 depending on how many rows. if 1 row, add as row, if 2+ rows, add as column"""

        """
        list_of_cts shape is (2, 3, 1, 501, 468) for num_rows, num_cols, num_overlays, overay_shape
        """
        if(len(list_cts2)>0):
            for i in range(0,len(list_cts2)):
                """ loop through each row and append a column"""
                list_cts[i].extend([list_cts2[i][0]]) # append a new col to row1
                list_labelmaps[i].extend([list_labelmaps2[i][0]])
                list_of_voxel_spacing[i].extend([list_of_voxel_spacing2[i][0]])

        
        if (num_images_per_region==1):
            append_horizontally = True
        print("generating montage")

        my_montage = Montage() 
        my_montage.execute(list_cts,list_labelmaps, out_file, in_overlay_alpha, len(list_cts), len(list_cts[0]), \
            window_width=window_width, window_level=window_level, resolution=resolution, list_of_voxel_spacing=list_of_voxel_spacing)      


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
                      metavar='<string>',type=int, default=900)                                                                    
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', type=int,default=-1024)                                                                 
    parser.add_option("--qc_partiallung", 
                      help='Set to true if want to QC partial lung labelmap images', 
                      dest="qc_partiallung", action='store_true')  
    parser.add_option("--qc_lunglobes", 
                      help='Set to true if want to QC  lung lobe images', 
                      dest="qc_lunglobes", action='store_true')  
    parser.add_option("--qc_regionstypes", 
                      help='Set to true if want to QC all the types and regions in a projection', 
                      dest="qc_regionstypes", action='store_true')  
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
    print ("Reading CT...")
    ct_array, ct_header = image_io.read_in_numpy(options.in_ct) 
    
    partial_array = None
    print ("Reading partial lung labelmap...")
    if(options.in_partial):
        partial_array, partial_header = image_io.read_in_numpy(options.in_partial) 
    
    lobe_array = None
    if(options.in_lobes):
        print ("Reading lobe...")
        lobe_array, lobe_header = image_io.read_in_numpy(options.in_lobes)               
        
    list_of_lung_qc = []
    if(options.qc_partiallung):
       list_of_lung_qc.append('leftLung') 
       list_of_lung_qc.append('rightLung')        
    if(options.qc_lunglobes):
       list_of_lung_qc.append('leftLungLobes') 
       list_of_lung_qc.append('rightLungLobes') 

    if(options.qc_regionstypes):
       list_of_lung_qc.append('all_regions') 
       list_of_lung_qc.append('all_types') 
              
    if(len(list_of_lung_qc) > 0):              
        print("generating lung qc for: ")
        print(list_of_lung_qc)
        spacing=ct_header['spacing']
        my_lung_qc = LungSegmentationQC()
        my_lung_qc.execute(ct_array, options.output_file, qc_requested=list_of_lung_qc, \
            num_images_per_region=options.num_images_per_region, in_partial=partial_array, \
            in_lobe=lobe_array, in_overlay_alpha=options.overlay_opacity, window_width=options.window_width, \
            window_level=options.window_level, resolution=options.output_image_resolution, spacing=spacing)

