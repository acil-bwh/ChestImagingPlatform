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

        
"""
For displaying color maps
"""                        
class QcColorConventionsManager:
            
    qc_background = colors.ColorConverter.to_rgba('black', alpha=0.0) 
    
    cmap = None
    norm = None
    qc_color_mapping = None
    
    
    @staticmethod
    def applyConvention(conventionId):
        if conventionId in QcColorConventionsManager.qc_color_mapping:
            return QcColorConventionsManager.qc_color_mapping[conventionId]
        else:
            return QcColorConventionsManager.qc_background

    @staticmethod
    def buildColorMap(overlay_alpha = 0.85):    
        
        QcColorConventionsManager.qc_color_mapping = {
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
        
                
                                
            
class ImageOverlay:
    """
    General purpose class that generates overlays given in input volume and a labelmap.
    
    Parameters 
    ----------   


    """    

    def __init__(self):
        pass
        #self.num_columns = num_columns
        
    def rotate_ct_images_for_display(self, image, axis):
        if(axis=='axial'):
            """ flip along Y axis then rotate -90 degres"""

            rotated_image_temp = np.fliplr(\
                image)
                
            rotated_image=scipy.ndimage.interpolation.rotate(\
                rotated_image_temp, 90.0)
        else:
            rotated_image=scipy.ndimage.interpolation.rotate(\
                image, 90.0)
                                   
        return rotated_image

    def rotate_labelmap_images_for_display(self, image, axis):
        if(axis=='axial'):
            print("rotating axial")
            rotated_image_temp=np.fliplr(\
                image)
                
            rotated_image=scipy.ndimage.interpolation.rotate(\
                rotated_image_temp, 90.0, prefilter=False,  order=0)               
        else:
            rotated_image=scipy.ndimage.interpolation.rotate(\
                image, 90.0, prefilter=False,  order=0)
                                   
        return rotated_image
        
    def get_overlay_bounds_from_volume_bounds(self, bounds_ct, bounds_lm, num_overlays, axis):
        """ get the slice numbers for an even distribution of images over the volume """

        bounds_list=[]
        if (axis=='sagittal'):
            slice_numbers = np.linspace(bounds_lm[0],bounds_lm[1],num_overlays+1, endpoint=False)[1:,]
            for i in range(0,np.shape(slice_numbers)[0]):
                bounds_list.append([int(slice_numbers[i]),int((slice_numbers[i]+1)),bounds_ct[2],\
                    bounds_ct[3],bounds_ct[4],bounds_ct[5]])
      
        if (axis=='axial'):
            slice_numbers = np.linspace(bounds_lm[4],bounds_lm[5],num_overlays+1, endpoint=False)[1:,]
            for i in range(0,np.shape(slice_numbers)[0]):
                bounds_list.append([bounds_ct[0],\
                    bounds_ct[1],bounds_ct[2],bounds_ct[3],int(slice_numbers[i]),int((slice_numbers[i]+1))])            

        if (axis=='coronal'):
            slice_numbers = np.linspace(bounds_lm[2],bounds_lm[3],num_overlays+1, endpoint=False)[1:,]
            for i in range(0,np.shape(slice_numbers)[0]):
                bounds_list.append([bounds_ct[0], bounds_ct[1],int(slice_numbers[i]),\
                    int((slice_numbers[i]+1)), bounds_ct[4],bounds_ct[5]])  
                        
        return bounds_list
        
           
    def get_ct_projection(self, in_ct, axis='axial'):
        """ clamp"""
        in_ct[in_ct<(-1024)]=-1024
        in_ct[in_ct>(100)]=100
        
        """ project along axis"""   
        if (axis=='sagittal'):  
            axis_val=0
        if (axis=='coronal'):  
            axis_val=1              
        if (axis=='axial'):  
            axis_val=2    
            
        projection = np.mean(in_ct, axis=axis_val)
        projection = self.rotate_ct_images_for_display(projection, axis)
                
        return(projection)
         
    def get_ct_overlay(self, in_ct1, in_ct2, in_mask, num_overlays, axis='axial', is_checkerboard = False):
        #img = (1.0 - alpha)*fixed + alpha*moving
    
        xmax, ymax,zmax = np.max(np.where(in_mask>0), 1)
        xmin, ymin,zmin = np.min(np.where(in_mask>0), 1)

        xmaxct, ymaxct,zmaxct = np.max(np.where(in_ct1>0), 1)
        xminct, yminct,zminct = np.min(np.where(in_ct1>0), 1)        

        bounds_lm = [xmin,xmax, ymin,ymax, zmin,zmax] 
        bounds_ct = [xminct, xmaxct, yminct,ymaxct, zminct,zmaxct ]
        
        bounds_list = self.get_overlay_bounds_from_volume_bounds(bounds_ct, bounds_lm, num_overlays, axis=axis)                                    

        """ Exract the list of overlays """
        list_of_cts = []

        print(bounds_list)
        for i in range(0,num_overlays):
            ct_patch1 = np.squeeze(in_ct1[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)
            
            ct_patch2 = np.squeeze(in_ct2[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)  
                
            ct_patch1 = self.rotate_ct_images_for_display(ct_patch1, axis)                  
            ct_patch2 = self.rotate_ct_images_for_display(ct_patch2, axis)  
            
            if(is_checkerboard):#list_of_overlays.append([ct_patch,lm_patch])
                import SimpleITK as sitk
                img1 = sitk.GetImageFromArray(ct_patch1)
                img2 = sitk.GetImageFromArray(ct_patch2)
                chck_xy = sitk.CheckerBoard(img1, img2, checkerPattern=[10,10])
                check_img = sitk.GetArrayFromImage(chck_xy)
                list_of_cts.append([check_img])
            else:    
                list_of_cts.append([ct_patch1,ct_patch2])

        return list_of_cts #overlay_images
        
                                             
    def get_segmentation_overlay(self, in_ct, in_lm, num_overlays, regions, axis='axial'):
 
        parser = RegionTypeParser(in_lm)

        mychestConvenstion =ChestConventions()
        reg_val =  mychestConvenstion.GetChestRegionValueFromName(regions[0])

        mask =  parser.get_mask(chest_region=reg_val) 
        for i in range(1,np.shape(regions)[0]):             
            reg_val =  mychestConvenstion.GetChestRegionValueFromName(regions[i])
            mask = np.logical_or(mask, parser.get_mask(chest_region=reg_val) ) 
        
        
        lm_masked = np.copy(in_lm)
        lm_masked[mask==0]=0 
        
        xmax, ymax,zmax = np.max(np.where(lm_masked>0), 1)
        xmin, ymin,zmin = np.min(np.where(lm_masked>0), 1)

        xmaxct, ymaxct,zmaxct = np.max(np.where(in_ct>0), 1)
        xminct, yminct,zminct = np.min(np.where(in_ct>0), 1)        

        bounds_lm = [xmin,xmax, ymin,ymax, zmin,zmax] 
        bounds_ct = [xminct, xmaxct, yminct,ymaxct, zminct,zmaxct ]
        
        bounds_list = self.get_overlay_bounds_from_volume_bounds(bounds_ct, bounds_lm, num_overlays, axis)                                    

        """ Exract the list of overlays """
        #list_of_overlays = []
        list_of_cts = []
        list_of_labelmaps = []

        print(bounds_list)
        for i in range(0,num_overlays):
            print(bounds_list[i])
            #pdb.set_trace()
            ct_patch = np.squeeze(in_ct[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)
            
            lm_patch = np.squeeze(in_lm[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)            

            ct_patch = self.rotate_ct_images_for_display(ct_patch, axis)
            lm_patch = self.rotate_labelmap_images_for_display(lm_patch, axis)
            #list_of_overlays.append([ct_patch,lm_patch])
            list_of_labelmaps.append([lm_patch])
            list_of_cts.append([ct_patch])
        #overlay_images = self.get_overlay_from_mask(in_ct, lm_masked, num_overlays, axis=axis) 
        
        return (list_of_cts, list_of_labelmaps) #overlay_images




"""
class for putting images together
"""                
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
        
    def __init__(self): #527,-993
        pass
 
       
    def execute(self, list_of_cts, list_of_labelmaps, output_filename, overlay_alpha, num_rows, num_columns, 
        window_width=1100, window_level=-1024, resolution=50):
        """ 
        for each image in list,
            add to montage. If image is empty, create an empty outline.
            
        Inputs
        ------
        list_of_input_images: list of num_rows*num_columns ndimages.
        
        output_filename: full path to output file to save the montage.
        """

        spec_val = np.max([num_rows,num_columns])
        print(spec_val) 
        fig = plt.figure(figsize = (spec_val,spec_val)) 
        gs=GridSpec(spec_val,spec_val, wspace=0.01, hspace=0.01)
        #gs.update(left=0.05, right=0.2) #, wspace=0.0, hspace=0.0) 
               
        QcColorConventionsManager.buildColorMap(overlay_alpha = overlay_alpha)
        from matplotlib.colors import LinearSegmentedColormap
        
        print(num_rows)
        print(num_columns)
        for i in range(0, num_rows):
            for j in range(0, num_columns):
                ax_1 = fig.add_subplot(gs[i:(i+1),j:(j+1)])
                #ax_1 = plt.Subplot(fig, gs[i*num_columns+j])

                #ax_1.axes.get_xaxis().set_visible(False)
                #ax_1.axes.get_yaxis().set_visible(False)
                ax_1.set_xticks([])
                ax_1.set_yticks([])
                ax_1.autoscale_view('tight')
                fig.add_subplot(ax_1)
                num_ct_overlays = len(list_of_cts[i][j])
                

                for k in range(0, num_ct_overlays):
                    list_of_cts[i][j][k][list_of_cts[i][j][k]<window_level] = window_level
                    list_of_cts[i][j][k][list_of_cts[i][j][k]>(window_level+window_width)] = (window_level+window_width)
                    list_of_cts[i][j][k] = list_of_cts[i][j][k]-np.min(list_of_cts[i][j][k])
                    list_of_cts[i][j][k] = list_of_cts[i][j][k]/float(np.max(list_of_cts[i][j][k]))  
                       


                if (num_ct_overlays > 1):
                        im_shape = np.shape(list_of_cts[i][j][0])

                        from PIL import Image
                        im1 = Image.new("RGB", [im_shape[1],im_shape[0]], "black")
                        im2 = Image.new("RGB", [im_shape[1],im_shape[0]], "black")
                        
                        im1_array = np.array(im1)
                        im2_array = np.array(im2) # im2_array = np.array(im2)
                        #rgb_arr2= np.zeros([im_shape[0], im_shape[1], 3]).astype('float')                          

                        im1_array[:,:,0] = list_of_cts[i][j][0]*255.0 
                        im2_array[:,:,2] = list_of_cts[i][j][1]*255.0 
                        
                        print(overlay_alpha)
                        image_blend = Image.blend(Image.fromarray(im1_array,'RGB'),Image.fromarray(im2_array,'RGB'),alpha=overlay_alpha)

                        
                        im = plt.imshow(np.array(image_blend),origin='upper')
                        #plt.tight_layout()

                else:
                      print("showing grey map output")  
                      myccmap= 'gray' #cm.gray
                      the_alpha = 1.0
                      #pdb.set_trace()  
                      im = plt.imshow(list_of_cts[i][j][0],origin='upper', \
                        cmap=plt.get_cmap(myccmap),  clim=(0.0, 1.0)) #0.2,0.8 was too much #vmin=0.2, vmax=0.8)
                      #plt.tight_layout()
                    
                    
        
                if((len(list_of_labelmaps) >0) and (len(list_of_labelmaps[i]) >0) and (len(list_of_labelmaps[i][j]) >0)):
                    """ append the overlay """
                    #list_of_labelmaps[i][j][0]=scipy.ndimage.interpolation.rotate(\
                    #list_of_labelmaps[i][j][0], 90.0, prefilter=False,  order=0) #reshape=False,
                    im=plt.imshow(list_of_labelmaps[i][j][0],origin='upper', \
                        cmap=QcColorConventionsManager.cmap , norm=QcColorConventionsManager.norm,\
                        vmin=1, vmax=100000) #norm=norm, alpha=0.95,

        #gs.tight_layout(fig)        
        print("saving to output file "+output_filename)
        #
        all_axes = fig.get_axes()

        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)
            #if ax.is_first_row():
            #    ax.spines['top'].set_visible(True)
            #if ax.is_last_row():
            #    ax.spines['bottom'].set_visible(True)
            #if ax.is_first_col():
            #    ax.spines['left'].set_visible(True)
            #if ax.is_last_col():
            #    ax.spines['right'].set_visible(True)


        #fig1 = plt.gcf()
        #gs.tight_layout(fig1)
        #fig1.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 0, 'h_pad': 0})
        plt.show()
        fig.savefig(output_filename, dpi=resolution,bbox_inches='tight', pad_inches=0.0)
        #fig1.clf() 
        #fig.clf()
        #plt.clf()

        #plt.close()    
 

"""
classes for generating specific types of QC
"""          
class LabelmapQC:
    """
    General purpose class that generates labelmap QC images.
    
    Takes as input a list of overlays, and for each overlay,
        takes in a list of required regions.

    """                       
    
    
    def __init__(self):
        pass
        
    
    def get_labelmap_qc(self, in_ct, out_file, list_of_labelmaps, list_request_qc_per_labelmap, num_images_per_region=3, 
            overlay_alpha=0.85, window_width=1100, window_level=-1024, axis='axial', resolution=600):   

    
        """ the layout of the images should be set here"""
        append_horizontally = False
            
        list_of_images = []
        list_cts = []
        list_labelmaps = []
        
        """ partial lung labelmap images"""
        for j in range (0, len(list_of_labelmaps)):
            for i in range(0, len(list_request_qc_per_labelmap[j])):
                #list_of_images.append([]) 
                in_regions = self.region_dictionary[list_request_qc_per_labelmap[j][i]]
                overlay = ImageOverlay()
                
                print("about to generate overlay "+ str(num_images_per_region)+" ")
                print(in_regions)
  
                #list_of_images[i] = overlay.execute( in_ct, list_of_labelmaps[j],num_images_per_region, in_regions) 
                temp_list_cts,temp_list_labelmaps = overlay.get_segmentation_overlay( in_ct, list_of_labelmaps[j],num_images_per_region, in_regions, axis=axis)

                list_cts.append(temp_list_cts)
                list_labelmaps.append(temp_list_labelmaps)
                #list_of_images.append(overlay.execute( in_ct, list_of_labelmaps[j],num_images_per_region, in_regions, axis=axis)) 

            if (num_images_per_region==1):
                append_horizontally = True
       
        print("generating montage")
        #pdb.set_trace()
        my_montage = montage() 
        my_montage.execute(list_cts,list_labelmaps, out_file, overlay_alpha, len(list_cts), num_images_per_region, \
            window_width=window_width, window_level=window_level, resolution=resolution)                

class CTQC:
    """
    General purpose class that generates CT QC images.
    """
    
    def __init__(self):
        pass
        
    
    def execute(self, in_ct, out_file, window_width=1100, window_level=-1024, resolution=50):

        my_projection = ImageOverlay() 
        
        x_projection = my_projection.get_ct_projection(in_ct, axis='sagittal')
        y_projection = my_projection.get_ct_projection(in_ct, axis='coronal')
        z_projection = my_projection.get_ct_projection(in_ct, axis='axial')

        # combine into 1 image
        my_montage = montage()
        
        #print(np.shape([[x_projection, y_projection, z_projection]]))
        #pdb.set_trace()
        my_montage.execute([[[x_projection], [y_projection], [z_projection]]],[], \
            out_file, 0.0, 1, 3, window_width=window_width, window_level=window_level,\
            resolution=resolution)
            
    def execute_registration_qc(self, in_ct1, in_ct2, out_file, in_mask, num_images, 
        overlay_alpha=0.85, window_width=1100, window_level=-1024, resolution=600):
        
        my_overlay = ImageOverlay()
        #pdb.set_trace()
        list_cts = []
        list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images))
                
        # Append the checkerboard image
        list_cts.append(my_overlay.get_ct_overlay( in_ct1, in_ct2, in_mask, num_images,is_checkerboard=True))
        
        my_montage = montage() 
        my_montage.execute(list_cts,[], out_file, overlay_alpha, len(list_cts), num_images, \
            window_width=window_width, window_level=window_level, resolution=resolution)  
            
                                                                                
class LungQC(LabelmapQC):
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
            overlay_alpha=0.85, window_width=1100, window_level=-1024, resolution=600):   
        
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
            axis='sagittal', resolution=resolution)

    
class body_composition_qc:
    pass    
    

        

               

if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input ct file name. ', dest='in_ct', metavar='<string>',
                      default=None)          
    parser.add_option('--in_ct_moving',
                      help='Input moving ct file name. ', dest='in_ct_moving', metavar='<string>',
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
    parser.add_option('--window_width',
                      help='intensity window width .  (optional)',  dest='window_width', 
                      metavar='<string>', default=1100)                         
                                           
    parser.add_option('--window_level',
                      help='intensity window level .  (optional)',  dest='window_level', 
                      metavar='<string>', default=-1024)                                                                 
    parser.add_option("--qc_ct", 
                      help='Set to true if want to QC CT images', dest="qc_ct", 
                      action='store_true')   
    parser.add_option("--qc_registeredct", 
                      help='Set to true if want to QC registered CT images', dest="qc_registeredct", 
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
    parser.add_option('--overlay_opacity',
                    help='Opacity of QC overlay (between 0 and 1)',  dest='overlay_opacity', 
                    metavar='<string>', type=float, default=0.85)   	                                                                            
    parser.add_option('--resolution',
                      help='Output image resolution.  (optional)',  dest='output_image_resolution', 
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
    
    """ CT QC"""
    if(options.qc_ct):
        my_ct_qc = CTQC()
        my_ct_qc.execute(ct_array, options.output_file,  \
            window_width=options.window_width, window_level=options.window_level, resolution=options.output_image_resolution)
    
    """ Registration QC"""
    if(options.qc_registeredct):
        my_ct_qc = CTQC()
        
        ct_array2, ct_header2 = image_io.read_in_numpy(options.in_ct_moving) 
        
        my_ct_qc.execute_registration_qc(ct_array, ct_array2, options.output_file, partial_array, options.num_images_per_region, \
            overlay_alpha=options.overlay_opacity, window_width=options.window_width, window_level=options.window_level, \
            resolution=options.output_image_resolution)
               
        
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
        my_lung_qc = LungQC()
        my_lung_qc.execute(ct_array, options.output_file, qc_requested=list_of_lung_qc, \
            num_images_per_region=options.num_images_per_region, in_partial=partial_array, \
            in_lobe=lobe_array, overlay_alpha=options.overlay_opacity, window_width=options.window_width, \
            window_level=options.window_level, resolution=options.output_image_resolution)

