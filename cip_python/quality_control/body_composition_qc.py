import numpy as np
from optparse import OptionParser
import warnings
from cip_python.input_output import ImageReaderWriter
from cip_python.quality_control.qc_base import LabelmapQC
from cip_python.quality_control.qc_base import Montage

class BodyCompositionQC(LabelmapQC):
    """
    General purpose class that generates lung segmentation QC images.
    
    Parameters 
    ---------- 
      
    Npne
        
    """
    
    def __init__(self):


        self.region_dictionary  = {'pecsSubcutaneousFat': None, 'visceralFat': None}   
   
        self.type_dictionary  = {\
        'pecsSubcutaneousFat':  None,\
        'visceralFat': None,\
                }    
                
        self.region_type_dictionary={\
            'pecsSubcutaneousFat': [['Left','PectoralisMinor'],['Right','PectoralisMinor'],\
            ['Left','PectoralisMajor'],['Right','PectoralisMajor'],\
            ['Left','SubcutaneousFat'],['Right','SubcutaneousFat']],\
            
            'visceralFat': [['Paravertebral','SubcutaneousFat'], ['Abdomen','VisceralFat'],\
            ['Paravertebral','Muscle'] ],\
        }        
        
        


        
        self.axes_dictionary = {\
        'pecsSubcutaneousFat': 'axial',\
        'visceralFat': 'axial',\
        }            
                                                
        self.labelmap_qc_dictionary={'bodyComposition':['pecsSubcutaneousFat','visceralFat']} #,'all_regions','all_types'
        
        self.overlay_qc = {'pecsSubcutaneousFat','visceralFat'}
        
     
        LabelmapQC.__init__(self)  
                             
    def execute(self, in_ct, out_file, qc_requested, num_images_per_region=3, in_body_composition=None,  \
            in_overlay_alpha=0.85, window_width=90, window_level=-50, resolution=600, spacing=None):   
        
        list_of_labelmaps_overlay=[] 
        list_request_qc_per_labelmap_overlay=[]
        list_axes_overlay=[]
        
        """ first check which labelmaps are available """
        available_labelmaps = []
        if in_body_composition is not None:
            available_labelmaps.append('bodyComposition')
                                                          
        """ add the labelmaps needed and perform labelmap QC"""                                                          
        for labelmaptype in available_labelmaps: 
            labelmap_qc_requested = set(self.labelmap_qc_dictionary[labelmaptype]).intersection(set(qc_requested)) 
            labelmap_qc_requested_overlay = labelmap_qc_requested.intersection(set(self.overlay_qc)) 

            if len(labelmap_qc_requested_overlay)>0:
                """ the labelmap tye is needed for some requested QC, thus add it to list"""
                if (labelmaptype=='bodyComposition'):
                    list_of_labelmaps_overlay.append(in_body_composition)
                   
                list_request_qc_per_labelmap_overlay.append(list(labelmap_qc_requested_overlay))       
                axes_temp=[]
                for qc in labelmap_qc_requested_overlay:
                    axes_temp.append(self.axes_dictionary[qc])
                list_axes_overlay.append(axes_temp)  

        [list_cts,list_labelmaps, list_of_voxel_spacing] = self.get_labelmap_qc(in_ct, out_file, list_of_labelmaps_overlay, \
            list_request_qc_per_labelmap_overlay, list_axes_overlay, num_images_per_region=num_images_per_region, \
            overlay_alpha=in_overlay_alpha, window_width=window_width, window_level=window_level,\
            resolution=resolution,spacing=spacing, is_overlay=True )
                        
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
    parser.add_option('--in_body_composition',
                      help='Input lung labelmap file name. If input \
                      lung labelmaps will be QCed, ', dest='in_body_composition', metavar='<string>',
                      default=None)                                                     
    parser.add_option('--window_width',
                      help='intensity window width .  (optional)',  dest='window_width', 
                      metavar='<string>',type=int, default=90)                                                                    
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', type=int,default=-50)                                                                 
    parser.add_option("--qc_pecs_subcutaneousfat", 
                      help='Set to true if want to QC pecs_subcutaneousfat images', 
                      dest="qc_pecs_subcutaneousfat", action='store_true')  
    parser.add_option("--qc_visceral_fat", 
                      help='Set to true if want to QC  qc_visceral_fat. Not yet functional.', 
                      dest="qc_visceral_fat", action='store_true')  
    parser.add_option("--qc_all_slices", 
                      help='Set to true if want to QC Every slice that has labels. Otherwise, it \
                      is assumed that only 1 slice has labels on it. Not yet functional.', 
                      dest="qc_all_slices", action='store_true')  
    parser.add_option("--qc_regionstypes", 
                      help='Set to true if want to QC all the types and regions in a projection. Not yet functional.', 
                      dest="qc_regionstypes", action='store_true')  
    parser.add_option('--output_file',
                      help='QC image output file name.', dest='output_file', metavar='<string>',
                      default=None)   
    parser.add_option('--overlay_opacity',
                    help='Opacity of QC overlay (between 0 and 1)',  dest='overlay_opacity', 
                    metavar='<string>', type=float, default=0.7)   	                                                                            
    parser.add_option('--resolution',
                      help='Output image resolution (dpi).  (optional)',  dest='output_image_resolution', 
                      metavar='<string>', type=float,  default=600)   	                                                                            	                                                                            	                                                                            
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    print ("Reading CT...")
    ct_array, ct_header = image_io.read_in_numpy(options.in_ct) 
    
    body_composition_array = None
    print ("Reading partial lung labelmap...")
    if(options.in_body_composition):
        body_composition_array, body_composition_header = image_io.read_in_numpy(options.in_body_composition) 
            
    list_of_bosycomposition_qc = []
    if(options.qc_pecs_subcutaneousfat):
       list_of_bosycomposition_qc.append('pecsSubcutaneousFat') 
    if(options.qc_visceral_fat):
       list_of_bosycomposition_qc.append('visceralFat') 
    if(options.qc_regionstypes):
       list_of_bosycomposition_qc.append('all_regions') 
       list_of_bosycomposition_qc.append('all_types') 
              
    if(len(list_of_bosycomposition_qc) > 0):              
        print("generating lung qc for: ")
        print(list_of_bosycomposition_qc)
        spacing=ct_header['spacing']
        my_lung_qc = BodyCompositionQC()
        
        if(options.qc_all_slices):
            z_slices=np.unique(np.where(body_composition_array>0)[2])
            for the_slice in z_slices:
                out_file=options.output_file.split(".")[0]+"_"+str(the_slice)+options.output_file.split(".")[1]
                my_lung_qc.execute(ct_array[:,:,the_slice], out_file, qc_requested=list_of_bosycomposition_qc, \
                num_images_per_region=1, in_body_composition=body_composition_array[:,:,the_slice], \
                in_overlay_alpha=options.overlay_opacity, window_width=options.window_width, \
                window_level=options.window_level, resolution=options.output_image_resolution, spacing=spacing)
        else:    
            my_lung_qc.execute(ct_array, options.output_file, qc_requested=list_of_bosycomposition_qc, \
                num_images_per_region=1, in_body_composition=body_composition_array, \
                in_overlay_alpha=options.overlay_opacity, window_width=options.window_width, \
                window_level=options.window_level, resolution=options.output_image_resolution, spacing=spacing)

