import numpy as np
from optparse import OptionParser
import warnings
#from cip_python.common import ChestConventions
from cip_python.input_output import ImageReaderWriter
from cip_python.utils import RegionTypeParser
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from cip_python.common import ChestConventions
import scipy
import matplotlib.pyplot as plt
from matplotlib import colors
import pdb 
        
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




    def __init__(self, num_overlays, regions):
        self.num_overlays = num_overlays
        self.regions = regions
        #self.num_columns = num_columns
        
        # build the color dictionary
     
    def get_overlay_from_mask(self, in_ct, in_lm):
        """ get the bounds for which in_lm >0 """
        
        print("starting overlay from mask")

        import pdb
        xmax, ymax,zmax = np.max(np.where(in_lm>0), 1)
        xmin, ymin,zmin = np.min(np.where(in_lm>0), 1)

        xmaxct, ymaxct,zmaxct = np.max(np.where(in_ct>0), 1)
        xminct, yminct,zminct = np.min(np.where(in_ct>0), 1)        

        bounds_lm = [xmin,xmax, ymin,ymax, zmin,zmax] 
        bounds_ct = [xminct, xmaxct, yminct,ymaxct, zminct,zmaxct ]
        """ get the slice numbers for an even distribution of images over the volume """
        
        slice_numbers = np.linspace(bounds_lm[0],bounds_lm[1],self.num_overlays+1, endpoint=False)[1:,]
        print(slice_numbers)
        #pdb.set_trace()
        """ Exract the list of overlays """
        list_of_overlays = []
        for i in range(0,np.shape(slice_numbers)[0]):
            ct_patch = np.squeeze(in_ct[int(slice_numbers[i]):int((slice_numbers[i]+1)),bounds_ct[2]:bounds_ct[3],bounds_ct[4]:bounds_ct[5]]).astype(float) # temporary
            lm_patch = np.squeeze(in_lm[int(slice_numbers[i]):int((slice_numbers[i]+1)),bounds_ct[2]:bounds_ct[3],bounds_ct[4]:bounds_ct[5]]).astype(float) # temporary

            list_of_overlays.append([ct_patch,lm_patch])

        """ return the list of overlays """
        return list_of_overlays    
        
    def execute(self, in_ct, in_lm):
 
        parser = RegionTypeParser(in_lm)

        mychestConvenstion =ChestConventions()
        reg_val =  mychestConvenstion.GetChestRegionValueFromName(self.regions[0])

        mask =  parser.get_mask(chest_region=reg_val) 
        for i in range(1,np.shape(self.regions)[0]):             
            reg_val =  mychestConvenstion.GetChestRegionValueFromName(self.regions[i])
            mask = np.logical_or(mask, parser.get_mask(chest_region=reg_val) ) 
        
        
        lm_masked = np.copy(in_lm)
        lm_masked[mask==0]=0 
        overlay_images = self.get_overlay_from_mask(in_ct, lm_masked) 
        
        return overlay_images


class QcColorConventionsManager:
    overlay_alpha = 0.85
    qc_color_mapping = {
        'LEFTSUPERIORLOBE':  colors.ColorConverter.to_rgba('red', alpha=overlay_alpha),
        'LEFTINFERIORLOBE': colors.ColorConverter.to_rgba('green', alpha=overlay_alpha),
        'RIGHTSUPERIORLOBE': colors.ColorConverter.to_rgba('cyan', alpha=overlay_alpha),
        'RIGHTMIDDLELOBE': colors.ColorConverter.to_rgba('purple', alpha=overlay_alpha),
        'RIGHTINFERIORLOBE': colors.ColorConverter.to_rgba('blue', alpha=overlay_alpha), # these above are like cip montage
        'LEFTUPPERTHIRD': colors.ColorConverter.to_rgba('cyan', alpha=overlay_alpha),
        'LEFTMIDDLETHIRD': colors.ColorConverter.to_rgba('purple', alpha=overlay_alpha),
        'LEFTLOWERTHIRD':colors.ColorConverter.to_rgba('green', alpha=overlay_alpha),
        'RIGHTMIDDLETHIRD': colors.ColorConverter.to_rgba('yellow', alpha=overlay_alpha),
        'RIGHTLOWERTHIRD': colors.ColorConverter.to_rgba('blue', alpha=overlay_alpha),
        'RIGHTUPPERTHIRD': colors.ColorConverter.to_rgba('red', alpha=overlay_alpha),
        }    
        
    qc_background = colors.ColorConverter.to_rgba('black', alpha=0.0) 
    
    cmap = None
    norm = None
    
    #LEFTSUPERIORLOBE='red'
    #LEFTINFERIORLOBE='green'
    #RIGHTSUPERIORLOBE='cyan'
    #RIGHTMIDDLELOBE='purple'
    #RIGHTINFERIORLOBE='blue' # these above are like cip montage
    #LEFTUPPERTHIRD='cyan'
    #LEFTMIDDLETHIRD='purple'
    #LEFTLOWERTHIRD='green'
    #RIGHTUPPERTHIRD='red'
    #RIGHTMIDDLETHIRD='yellow'
    #RIGHTLOWERTHIRD='blue'
    
    @staticmethod
    def applyConvention(conventionId):
        if conventionId in QcColorConventionsManager.qc_color_mapping:
            return QcColorConventionsManager.qc_color_mapping[conventionId]
        else:
            return QcColorConventionsManager.qc_background

    @staticmethod
    def buildColorMap():         
        color_bound_list = [[QcColorConventionsManager.qc_background, 0 ]]
        
        mychestConvenstion =ChestConventions()
        for region in QcColorConventionsManager.qc_color_mapping.keys():
            """ append the color to the color list"""
            """ get the label bounds and append to bounds list. The bounds are lblval:lblval+1"""
            reg_val =  mychestConvenstion.GetChestRegionValueFromName(region)   
            color_bound_list.append([QcColorConventionsManager.qc_color_mapping[region], reg_val])        
        
        """ Sort the tuples and turn back to list """
        color_bound_list.sort(key=lambda tup: tup[1]) 
        color_bound_list = zip(*color_bound_list)
        color_bound_list[0]=list(color_bound_list[0])
        color_bound_list[1]=list(color_bound_list[1])
        """ Go through list sequentially and add bounds if non-existent"""
        for i,x in enumerate(color_bound_list[1][:-1]):
            if ((color_bound_list[1][i+1] != color_bound_list[1][i]+1) and (color_bound_list[0][i] != QcColorConventionsManager.qc_background)):
                color_bound_list[0].insert(i+1, QcColorConventionsManager.qc_background)
                color_bound_list[1].insert(i+1, color_bound_list[1][i]+1)

        color_bound_list[0].append(QcColorConventionsManager.qc_background)
        color_bound_list[1].append(color_bound_list[1][-1]+1)              
        color_bound_list[0].append(QcColorConventionsManager.qc_background)
        color_bound_list[1].append(color_bound_list[1][-1]+1)
        QcColorConventionsManager.cmap = colors.ListedColormap(color_bound_list[0]) 
        QcColorConventionsManager.norm = colors.BoundaryNorm(color_bound_list[1], QcColorConventionsManager.cmap.N)

        
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
        
    def __init__(self, num_rows, num_columns, window_width=1100, window_level=-1024): #527,-993
        self.num_rows = num_rows # Not necessary
        self.num_columns = num_columns
        self.window_width=window_width
        self.window_level=window_level
 
       
    def execute(self, list_of_input_images, output_filename):
        """ 
        for each image in list,
            add to montage. If image is empty, create an empty outline.
            
        Inputs
        ------
        list_of_input_images: list of num_rows*num_columns ndimages.
        
        output_filename: full path to output file to save the montage.
        """
        
        fig = plt.figure() 
        gs=GridSpec(self.num_rows,self.num_columns)
                
        QcColorConventionsManager.buildColorMap()
        
        for i in range(0, self.num_rows):
            for j in range(0, self.num_columns):
                ax_1 = fig.add_subplot(gs[i:(i+1),j:(j+1)])
                ax_1.axes.get_xaxis().set_visible(False)
                ax_1.axes.get_yaxis().set_visible(False)

                # check if there are 1 or 2 images per entry. if 1: no overlay required.
                
                list_of_input_images[i][j][0][list_of_input_images[i][j][0]<self.window_level] = self.window_level
                list_of_input_images[i][j][0][list_of_input_images[i][j][0]>(self.window_level+self.window_width)] = (self.window_level+self.window_width)
                list_of_input_images[i][j][0] = list_of_input_images[i][j][0]-np.min(list_of_input_images[i][j][0])
                list_of_input_images[i][j][0] = list_of_input_images[i][j][0]/float(np.max(list_of_input_images[i][j][0]))      

                
                import pdb
                transform=np.array([[0,1],[1,0]]) # Need a 90 degree rotation

                list_of_input_images[i][j][0]=scipy.ndimage.interpolation.rotate(\
                    list_of_input_images[i][j][0], 90.0)

                im = plt.imshow(list_of_input_images[i][j][0],origin='upper', \
                    cmap=cm.gray,  clim=(0.0, 1.0)) #0.2,0.8 was too much #vmin=0.2, vmax=0.8)
                plt.tight_layout()
        
                if(np.shape(list_of_input_images[i][j])[0] >1):
                    """ append the overlay """
                    list_of_input_images[i][j][1]=scipy.ndimage.interpolation.rotate(\
                    list_of_input_images[i][j][1], 90.0, prefilter=False,  order=0) #reshape=False,
                    plt.imshow(list_of_input_images[i][j][1],origin='upper', \
                        cmap=QcColorConventionsManager.cmap , norm=QcColorConventionsManager.norm,\
                        vmin=1, vmax=100000) #norm=norm, alpha=0.95,

        gs.tight_layout(fig)        
        print("saving to output file "+output_filename)
        fig1 = plt.gcf()
        gs.tight_layout(fig1)
        fig.savefig(output_filename,bbox_inches='tight', dpi=300)
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
            list_of_images.append([]) 
            in_regions = self.region_dictionary[self.qc_requested[i]]
            overlay = image_overlay(self.num_images_per_region, in_regions)
  
            if ((self.qc_requested[i] == 'leftLung') or (self.qc_requested[i] == 'rightLung')):
                list_of_images[i] = overlay.execute( in_ct, in_partial) 

            if ((self.qc_requested[i] == 'leftLungLobes') or (self.qc_requested[i] == 'rightLungLobes')):
                list_of_images[i] = overlay.execute( in_ct, in_lobe) 

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
                      metavar='<string>', default=3)                        
    parser.add_option("--qc_ct", 
                      help='Set to true if want to QC CT images', dest="ct_qc", 
                      action='store_true')   
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
                      help='Prefix used for output QC images', dest='output_file', metavar='<string>',
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
        my_lung_qc = lung_qc(list_of_lung_qc, options.num_images_per_region)
        my_lung_qc.execute(ct_array, options.output_file, in_partial=partial_array, in_lobe=lobe_array)
