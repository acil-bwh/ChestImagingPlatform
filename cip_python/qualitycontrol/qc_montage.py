import numpy as np
from optparse import OptionParser
import warnings
#from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec



class projection_image:
    pass
    
    
class image_overlay:
    """
    General purpose class that generates overlays given in input volume and a labelmap.
    
    Parameters 
    ----------   
    num_overlays: int
        number of num_overlays to generate. The overlays will be evently spaced
        in the z axis of the volume. If 1 overlay is requested, it will
        be generated in the middle of the volume.

    """    

    qc_color_mapping = {
        'LEFTSUPERIORLOBE': 'red', 
        'LEFTINFERIORLOBE': 'green',
        'RIGHTSUPERIORLOBE': 'cyan',
        'RIGHTMIDDLELOBE': 'purple',
        'RIGHTINFERIORLOBE': 'blue'
        #'LEFTSUPERIORLUNG': ...
        #default: 'black'}
        }

    def __init__(self, num_overlays, regions):
        self.num_overlays = num_overlays
        self.regions = regions
        #self.num_columns = num_columns
     
    def get_overlay_from_mask(self, in_ct, in_mask):
        """ get the bounds for which in_lm >0 """
        
        bounds = [0,np.shape(in_ct)[0], 0,np.shape(in_ct)[1], 0,np.shape(in_ct)[2]] # temporary
        
        """ get the slice numbers for an even distribution of images over the volume """
        slice_numbers = np.linspace(bounds[5],bounds[6],self.num_overlays+1, endpoint=False)

        """ Exract the list of overlays """
        list_of_overlays = []
        for i in range(0,np.shape(slice_numbers)[0]):
            full_slice = np.squeeze(in_ct[:,:,slice_numbers[i]:(slice_numbers[i]+1)]).astype(float) # temporary
            list_of_overlays.append(full_slice)

        """ return the list of overlays """
        return list_of_overlays    
        
    def execute(self, in_ct, in_lm):
 
        parser = RegionTypeParser(in_lm)
        mask =  parser.get_mask(chest_region=self.regions[0]) 
        for i in range(1,np.shape(self.regions)[0]):  
            mask = np.logical_or(mask, parser.get_mask(chest_region=self.regions[i]) )        
            #overlay = image_overlay(self.num_images_per_region, in_regions)
            overlay_images = self.get_overlay_from_mask(in_ct, mask) 
        
        return overlay_images

class montage:
    
    """
    General purpose class that generates a montage of images for QC purposes.
    
    Parameters 
    ----------   
    num_rows: int
        number of rows of images in the montage
    num_cols: int
        number of columns of images in the montage
    """
    
    def __init__(self, num_rows, num_columns):
        self.num_rows = num_rows # Not necessary
        self.num_columns = num_columns
        
    def execute(self, list_of_input_images, output_filename):
        """ 
        for each image in list,
            add to montage. If image is empty, create an empty outline.
            
        Inputs
        ------
        list_of_input_images: list of num_rows*num_columns ndimages.
        
        output_filename: full path to output file to save the montage.
        """
        
        import pyplot as plt
        fig = plt.figure() 
        gs=GridSpec(self.num_rows,self.num_columns)
        
        for i in range(0, self.num_rows):
            for j in range(0, self.num_columns):
                ax_1 = fig.add_subplot(gs[i:(i+1),j:(j+1)])
                ax_1.axes.get_xaxis().set_visible(False)
                ax_1.axes.get_yaxis().set_visible(False)
        
                im = plt.imshow(list_of_input_images[i][j],origin='upper', \
                    cmap=cm.gray, vmin=0., vmax=1.0)
        
            fig1 = plt.gcf()
            fig1.savefig(output_filename,bbox_inches='tight', dpi=300)
            fig1.clf() 
            fig.clf()
            plt.clf()

            plt.close()    
            
class lung_qc:
    """
    General purpose class that generates lung segmentation QC images.
    
    Parameters 
    ---------- 
      
    qc_requested: list 
          the regions for which the qc is required. Example:
          leftLung, rightLung, leftLungLobes, rightLungLobes    
            
    num_images_per_region: int
        number of rows of images in the montage
        
    """
    
    def __init__(self, qc_requested, num_images_per_region):
        self.qc_requested = qc_requested
        self.num_images_per_region = num_images_per_region

        self.region_dictionary  = {\
        'leftLung': ['LEFTUPPERTHIRD','LEFTMIDDLETHIRD','LEFTLOWERTHIRD'],\
        'rightLung': ['RIGHTUPPERTHIRD','RIGHTMIDDLETHIRD','RIGHTLOWERTHIRD'],\
        'leftLungLobes': ['LEFTSUPERIORLOBE','LEFTINFERIORLOBE'],\
        'rightLungLobes' : ['RIGHTSUPERIORLOBE', 'RIGHTMIDDLELOBE',\
                'RIGHTINFERIORLOBE'] 
                }   
     

                             
    def execute(self, in_ct, out_file, in_partial=None, in_lobe=None):   
        """ if the num_images_per_region is 1, then every qc_requested is appended horizontally.
            Else, every qc_requested is appended vertically. """ 
    
        """ the layout of the images should be set here"""
        append_horizontally = False

            
        list_of_images = []
        
        """ partial lung labelmap images"""
        for i in range(0, len(self.qc_requested)):
                            #list_of_images.append([]) 
            in_regions = self.region_dictionary[self.qc_requested[i]]
            overlay = image_overlay(self.num_images_per_region, in_regions)
  
            if ((self.qc_requested[i] == 'leftLung') or (self.qc_requested[i] == 'rightLung')):
                list_of_images[i] = overlay.execute(self, in_ct, in_partial) 

            if ((self.qc_requested[i] == 'leftLungLobes') or (self.qc_requested[i] == 'rightLungLobes')):
                list_of_images[i] = overlay.execute(self, in_ct, in_partial) 

        if (self.num_images_per_region==1):
            append_horizontally = True
       
        my_montage = montage(len(self.qc_requested), self.num_images_per_region) 
        my_montage.execute(list_of_images, out_file)                


    
        
    # QualityControl.cxx

    
class body_composition_qc:
    pass    
    
class ct_qc:
    """
    General purpose class that generates CT QC images.
    
    """



if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
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
                      help='Number of images .  (optional)',  dest='num_images_per_region', 
                      metavar='<string>', default=31)                        
    parser.add_argument("--qc_ct", 
                      help='Set to true if want to QC CT images', dest="ct_qc", 
                      action='store_true')   
    parser.add_argument("--qc_leftlung", 
                      help='Set to true if want to QC left partial lung labelmap images', 
                      dest="qc_leftlung", action='store_true')  
    parser.add_argument("--qc_rightlung", 
                      help='Set to true if want to QC right partial lung labelmap images', 
                      dest="qc_rightlung", action='store_true')  
    parser.add_argument("--qc_leftlunglobes", 
                      help='Set to true if want to QC left lung lobe images', 
                      dest="qc_leftlunglobe", action='store_true')  
    parser.add_argument("--qc_rightlunglobes", 
                      help='Set to true if want to QC right lung lobe images', 
                      dest="qc_rightlunglobes", action='store_true')  
    parser.add_option('--output_prefix',
                      help='Prefix used for output QC images', dest='output_prefix', metavar='<string>',
                      default=None)   
	                                                                            
	                                                                            	                                                                            	                                                                            
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    print "Reading CT..."
    ct_array, ct_header = image_io.read_in_numpy(options.in_ct) 
    
    partial_array = None
    print "Reading partial lung labelmap..."
    if(options.in_partial):
        partial_array, partial_header = image_io.read_in_numpy(options.in_partial) 
    
    lobe_array = None
    if(options.in_lobe):
        print "Reading lobe..."
        lobe_array, lobe_header = image_io.read_in_numpy(options.in_lobes) 
    
    list_of_lung_qc = []
    if(options.qc_leftlung):
       list_of_lung_qc.append('leftLung') 
    if(options.qc_leftlung):
       list_of_lung_qc.append('rightLung') 
    if(options.qc_leftlunglobes):
       list_of_lung_qc.append('leftLungLobes') 
    if(options.qc_rightlunglobes):
       list_of_lung_qc.append('rightLungLobes') 

    if(len(list_of_lung_qc) > 0):              
        my_lung_qc = lung_qc(list_of_lung_qc, options.num_images_per_region)