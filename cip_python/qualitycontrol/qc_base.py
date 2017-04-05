import numpy as np
import warnings
#from cip_python.common import ChestConventions
from cip_python.utils import RegionTypeParser
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from cip_python.common import ChestConventions
import scipy
import matplotlib.pyplot as plt
from matplotlib import colors

        
"""
Class for displaying color maps
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
        list_of_cts = []
        list_of_labelmaps = []

        for i in range(0,num_overlays):
            ct_patch = np.squeeze(in_ct[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)
            
            lm_patch = np.squeeze(in_lm[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)            

            ct_patch = self.rotate_ct_images_for_display(ct_patch, axis)
            lm_patch = self.rotate_labelmap_images_for_display(lm_patch, axis)
            list_of_labelmaps.append([lm_patch])
            list_of_cts.append([ct_patch])
        
        return (list_of_cts, list_of_labelmaps) 




"""
class for putting images together
"""                
class Montage:
    
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
        fig = plt.figure(figsize = (spec_val,spec_val)) 
        gs=GridSpec(spec_val,spec_val, wspace=0.01, hspace=0.01)
               
        QcColorConventionsManager.buildColorMap(overlay_alpha = overlay_alpha)
        from matplotlib.colors import LinearSegmentedColormap
        
        for i in range(0, num_rows):
            for j in range(0, num_columns):
                ax_1 = fig.add_subplot(gs[i:(i+1),j:(j+1)])

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
                        im2_array = np.array(im2) 

                        im1_array[:,:,0] = list_of_cts[i][j][0]*255.0 
                        im2_array[:,:,2] = list_of_cts[i][j][1]*255.0 
                        
                        image_blend = Image.blend(Image.fromarray(im1_array,'RGB'),Image.fromarray(im2_array,'RGB'),alpha=overlay_alpha)

                        
                        im = plt.imshow(np.array(image_blend),origin='upper')

                else:
                      myccmap= 'gray' 
                      the_alpha = 1.0

                      im = plt.imshow(list_of_cts[i][j][0],origin='upper', \
                        cmap=plt.get_cmap(myccmap),  clim=(0.0, 1.0)) #0.2,0.8 was too much #vmin=0.2, vmax=0.8)
                     
        
                if((len(list_of_labelmaps) >0) and (len(list_of_labelmaps[i]) >0) and (len(list_of_labelmaps[i][j]) >0)):
                    """ append the overlay """

                    im=plt.imshow(list_of_labelmaps[i][j][0],origin='upper', \
                        cmap=QcColorConventionsManager.cmap , norm=QcColorConventionsManager.norm,\
                        vmin=1, vmax=100000) #norm=norm, alpha=0.95,

        print("saving to output file "+output_filename)

        all_axes = fig.get_axes()

        #show only the outside spines
        for ax in all_axes:
            for sp in ax.spines.values():
                sp.set_visible(False)

        plt.show()
        fig.savefig(output_filename, dpi=resolution,bbox_inches='tight', pad_inches=0.0)

 

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
                
                print("generating overlay for the following regions: ")
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
        my_montage = Montage() 
        my_montage.execute(list_cts,list_labelmaps, out_file, overlay_alpha, len(list_cts), num_images_per_region, \
            window_width=window_width, window_level=window_level, resolution=resolution)      