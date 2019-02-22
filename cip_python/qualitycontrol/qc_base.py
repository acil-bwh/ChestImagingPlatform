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
import pdb
        
"""
Class for displaying color maps
"""                        
class QcColorConventionsManager:
            
    my_colors = colors.ColorConverter()        
    qc_background = my_colors.to_rgba('black', alpha=0.0) 
    
    cmap = None
    norm = None
    qc_color_mapping = None
    
    
    #@staticmethod
    #def applyConvention(conventionId):
    #    if conventionId in QcColorConventionsManager.qc_color_mapping:
    #        return QcColorConventionsManager.qc_color_mapping[conventionId]
    #    else:
    #        return QcColorConventionsManager.qc_background

    @staticmethod
    def buildColorMap(overlay_alpha = 0.85):    
        
        my_colors = colors.ColorConverter()
        qc_allregionstypes = my_colors.to_rgba('brown', alpha=overlay_alpha) 

    
        QcColorConventionsManager.qc_color_mapping = {
        #'UNDEFINEDREGION':(0.00,0.00,0.00,0.0),

        'LEFTSUPERIORLOBE':  my_colors.to_rgba('red', alpha=overlay_alpha),
        'LEFTINFERIORLOBE': my_colors.to_rgba('green', alpha=overlay_alpha),
        'RIGHTSUPERIORLOBE': my_colors.to_rgba('cyan', alpha=overlay_alpha),
        'RIGHTMIDDLELOBE': my_colors.to_rgba('purple', alpha=overlay_alpha),
        'RIGHTINFERIORLOBE': my_colors.to_rgba('blue', alpha=overlay_alpha), # these above are like cip montage
        'LEFTUPPERTHIRD': my_colors.to_rgba('cyan', alpha=overlay_alpha),
        'LEFTMIDDLETHIRD': my_colors.to_rgba('purple', alpha=overlay_alpha),
        'LEFTLOWERTHIRD': my_colors.to_rgba('green', alpha=overlay_alpha),
        'RIGHTMIDDLETHIRD': my_colors.to_rgba('yellow', alpha=overlay_alpha),
        'RIGHTLOWERTHIRD': my_colors.to_rgba('blue', alpha=overlay_alpha),
        'RIGHTUPPERTHIRD': my_colors.to_rgba('red', alpha=overlay_alpha),
        'LEFT':(0.45,0.00,0.00,overlay_alpha),
        'RIGHT':(0.00,0.45,0.00,overlay_alpha),       
        # copied over from chest conventions
        'WHOLELUNG':(0.42,0.38,0.75,overlay_alpha),
        'RIGHTLUNG':(.26,0.64,0.10,overlay_alpha),
        'LEFTLUNG':(0.80,0.11,0.36,overlay_alpha),
        'MEDIASTINUM':(0.00,0.06,0.00,overlay_alpha),
        'WHOLEHEART':(0.00,0.07,0.00,overlay_alpha),
        'AORTA':(0.00,0.08,0.00,overlay_alpha),
        'PULMONARYARTERY':(0.00,0.09,0.00,overlay_alpha),
        'PULMONARYVEIN':(0.00,0.00,0.01,overlay_alpha),
        'UPPERTHIRD':(0.00,0.00,0.02,overlay_alpha),
        'MIDDLETHIRD':(0.00,0.00,0.03,overlay_alpha),
        'LOWERTHIRD':(0.00,0.00,0.04,overlay_alpha),
        'LIVER':(0.66,0.36,0.40,overlay_alpha),
        'SPLEEN':(1.00,1.00,0.01,overlay_alpha),
        'ABDOMEN':(1.00,0.50,0.01,overlay_alpha),
        'PARAVERTEBRAL':(1.00,0.51,0.01,overlay_alpha),
        'OUTSIDELUNG':(0.49,0.49,0.49,overlay_alpha),
        'OUTSIDEBODY':(0.49,0.49,0.50,overlay_alpha),
        'SKELETON':(0.49,0.49,0.51,overlay_alpha),
        'STERNUM':(0.49,0.49,0.52,overlay_alpha),
        'HUMERI':(0.49,0.49,0.53,overlay_alpha),
        'LEFTHUMERUS':(0.49,0.49,0.54,overlay_alpha),
        'RIGHTHUMERUS':(0.49,0.49,0.55,overlay_alpha),
        'SCAPULAE':(0.49,0.49,0.56,overlay_alpha),
        'LEFTSCAPULA':(0.49,0.49,0.57,overlay_alpha),
        'RIGHTSCAPULA':(0.49,0.49,0.58,overlay_alpha),
        'HILA':(0.49,0.49,0.59,overlay_alpha),
        'LEFTHILUM':(0.49,0.49,0.60,overlay_alpha),
        'RightHilum':(0.49,0.49,0.61,overlay_alpha),
        'KIDNEYS':(0.49,0.49,0.62,overlay_alpha),
        'LEFTKIDNEY':(0.49,0.49,0.63,overlay_alpha),
        'RIGHTKIDNEY':(0.49,0.49,0.64,overlay_alpha),
        'ASCENDINGAORTA':(0.49,0.49,0.65,overlay_alpha),
        'TRANSVERSALAORTA':(0.49,0.49,0.66,overlay_alpha),
        'DESCENDINGAORTA':(0.49,0.49,0.68,overlay_alpha),
        'LEFTSUBCLAVIAN':(00.49,0.49,0.69,overlay_alpha),
        'RIGHTSUBCLAVIAN':(00.49,0.49,0.70,overlay_alpha),
        'LEFTCORONARYARTERY':(00.49,0.49,0.71,overlay_alpha),
        'SPINE':(00.49,0.49,0.72,overlay_alpha),
        'LEFTVENTRICLE':(00.49,0.49,0.73,overlay_alpha),
        'RIGHTVENTRICLE':(00.49,0.49,0.74,overlay_alpha),
        'LEFTATRIUM':(00.49,0.49,0.75,overlay_alpha),
        'RIGHTATRIUM':(00.49,0.49,0.76,overlay_alpha),
        'LEFTPECTORALIS':(00.49,0.49,0.77,overlay_alpha),
        'RIGHTPECTORALIS':(00.49,0.49,0.78,overlay_alpha),
        'TRACHEA2':(00.49,0.49,0.79,overlay_alpha),
        'LEFTMAINBRONCHUS':(00.49,0.49,0.80,overlay_alpha),
        'RIGHTMAINBRONCHUS':(00.49,0.49,0.81,overlay_alpha),
        'ESOPHAGUS':(00.49,0.49,0.82,overlay_alpha),
        'LEFTCHESTWALL':(00.49,0.49,0.83,overlay_alpha),
        'RIGHTCHESTWALL':(00.49,0.49,0.84,overlay_alpha),
        'LEFTDIAPHRAGM':(00.49,0.49,0.85,overlay_alpha),
        'RIGHTDIAPHRAGM':(0>0.49,0.49,0.86,overlay_alpha),
        'HIATUS':(00.49,0.49,0.87,overlay_alpha),
        'PECTORALIS':(00.49,0.49,0.88,overlay_alpha),
        'SPINALCORD':(00.49,0.49,0.89,overlay_alpha),
        'SUPERIORMESENTERICARTERY':(00.49,0.49,0.9,overlay_alpha),
        'PANCREAS':(00.49,0.49,0.91,overlay_alpha),
        #'PANCREASHEAD':(00.49,0.49,0.92,overlay_alpha),
        'LEFTANTERIORCHESTWALL':(00.49,0.49,0.835,overlay_alpha),
        'RIGHTANTERIORCHESTWALL':(00.49,0.49,0.845,overlay_alpha),
        'LEFTPOSTERIORCHESTWALL':(00.49,0.49,0.8355,overlay_alpha),
        'RIGHTPOSTERIORCHESTWALL':(00.49,0.49,0.8455,overlay_alpha),
        'TRACHEACARINA':(00.49,0.49,0.93,overlay_alpha),
        'AORTICVALVE':(00.49,0.9,0.93,overlay_alpha),
        }  
          
        QcColorConventionsManager.qc_type_color_mapping = {
        #'UNDEFINEDTYPE':(0.00,0.00,0.00,0.0),

        'Airway': my_colors.to_rgba('fuchsia', alpha=overlay_alpha),
        'Vessel': my_colors.to_rgba('limegreen', alpha=overlay_alpha),
        'OBLIQUEFISSURE': my_colors.to_rgba('salmon', alpha=overlay_alpha),        
        'HORIZONTALFISSURE': my_colors.to_rgba('lightslategray', alpha=overlay_alpha),

        'PECTORALISMINOR':(0.00,0.99,0.00,overlay_alpha),
        'PECTORALISMAJOR':(0.99,0.01,0.00,overlay_alpha),
        'SUBCUTANEOUSFAT':(0.99,0.01,0.99,overlay_alpha),
        'NORMALPARENCHYMA':(0.99,0.99,0.99,overlay_alpha),
        'EMPHYSEMATOUS':(0.96,0.96,0.96,overlay_alpha),
        'GROUNDGLASS':(0.95,0.95,0.95,overlay_alpha),
        'RETICULAR':(0.94,0.94,0.94,overlay_alpha),
        'NODULAR':(0.93,0.93,0.93,overlay_alpha),
        'MILDPARASEPTALEMPHYSEMA':(0.90,0.90,0.90,overlay_alpha),
        'MODERATEPARASEPTALEMPHYSEMA':(0.89,0.89,0.89,overlay_alpha),
        'SEVEREPARASEPTALEMPHYSEMA':(0.88,0.88,0.88,overlay_alpha),
        'MILDBULLA':(0.87,0.87,0.87,overlay_alpha),
        'MODERATEBULLA':(0.86,0.86,0.86,overlay_alpha),
        'SEVEREBULLA':(0.85,0.85,0.85,overlay_alpha),
        'MILDCENTRILOBULAREMPHYSEMA':(0.84,0.84,0.84,overlay_alpha),
        'MODERATECENTRILOBULAREMPHYSEMA':(0.83,0.83,0.83,overlay_alpha),
        'SEVERECENTRILOBULAREMPHYSEMA':(0.82,0.82,0.82,overlay_alpha),
        'MILDPANLOBULAREMPHYSEMA':(0.81,0.81,0.81,overlay_alpha),
        'MODERATEPANLOBULAREMPHYSEMA':(0.80,0.70,0.80,overlay_alpha),
        'SEVEREPANLOBULAREMPHYSEMA':(0.79,0.79,0.79,overlay_alpha),
        'AIRWAYWALLTHICKENING':(0.78,0.78,0.78,overlay_alpha),
        'AIRWAYCYLINDRICALDILATION':(0.77,0.77,0.77,overlay_alpha),
        'VARICOSEBRONCHIECTASIS':(0.76,0.76,0.76,overlay_alpha),
        'CYSTICBRONCHIECTASIS':(0.75,0.75,0.75,overlay_alpha),
        'CENTRILOBULARNODULE':(0.74,0.74,0.74,overlay_alpha),
        'MOSAICING':(0.73,0.73,0.73,overlay_alpha),
        'EXPIRATORYMALACIA':(0.72,0.72,0.72,overlay_alpha),
        'SABERSHEATH':(0.71,0.71,0.71,overlay_alpha),
        'OUTPOUCHING':(0.70,0.70,0.70,overlay_alpha),
        'MUCOIDMATERIAL':(0.69,0.69,0.69,overlay_alpha),
        'PATCHYGASTRAPPING':(0.68,0.68,0.68,overlay_alpha),
        'DIFFUSEGASTRAPPING':(0.67,0.67,0.67,overlay_alpha),
        'LINEARSCAR':(0.66,0.66,0.66,overlay_alpha),
        'CYST':(0.65,0.65,0.65,overlay_alpha),
        'ATELECTASIS':(0.64,0.64,0.64,overlay_alpha),
        'HONEYCOMBING':(0.63,0.63,0.63,overlay_alpha),
        'TRACHEA':(0.51,0.50,0.50,overlay_alpha),
        'MAINBRONCHUS':(0.55,0.27,0.07,overlay_alpha),
        'UPPERLOBEBRONCHUS':(1.00,0.65,0.00,overlay_alpha),
        'AIRWAYGENERATION3':(1.00,1.00,0.01,overlay_alpha),
        'AIRWAYGENERATION4':(1.00,0.01,1.00,overlay_alpha),
        'AIRWAYGENERATION5':(0.51,1.00,0.00,overlay_alpha),
        'AIRWAYGENERATION6':(0.01,0.50,1.00,overlay_alpha),
        'AIRWAYGENERATION7':(0.51,0.00,0.50,overlay_alpha),
        'AIRWAYGENERATION8':(0.51,0.50,0.00,overlay_alpha),
        'AIRWAYGENERATION9':(0.01,0.50,0.50,overlay_alpha),
        'AIRWAYGENERATION10':(0.45,0.44,0.44,overlay_alpha),
        'CALCIFICATION':(0.51,0.51,0.51,overlay_alpha),
        'ARTERY':(0.00,0.00,0.99,overlay_alpha),
        'VEIN':(0.99,0.00,0.00,overlay_alpha),
        'ANTERIORSCALENE':(0.46,0.46,0.46,overlay_alpha),
        'FISSURE':(0.93,0.01,0.10,overlay_alpha),
        'VESSELGENERATION0':(0.00,0.00,0.00,overlay_alpha),
        'VESSELGENERATION1':(0.00,1.00,0.00,overlay_alpha),
        'VESSELGENERATION2':(0.00,1.00,1.00,overlay_alpha),
        'VESSELGENERATION3':(1.00,1.00,0.00,overlay_alpha),
        'VESSELGENERATION4':(1.00,0.00,1.00,overlay_alpha),
        'VESSELGENERATION5':(0.50,1.00,0.00,overlay_alpha),
        'VESSELGENERATION6':(0.00,0.50,1.00,overlay_alpha),
        'VESSELGENERATION7':(0.50,0.00,0.50,overlay_alpha),
        'VESSELGENERATION8':(0.50,0.50,0.00,overlay_alpha),
        'VESSELGENERATION9':(0.00,0.50,0.50,overlay_alpha),
        'VESSELGENERATION10':(0.44,0.44,0.44,overlay_alpha),
        'PARASEPTALEMPHYSEMA':(0.00,0.68,0.00,overlay_alpha),
        'CENTRILOBULAREMPHYSEMA':(0.00,0.69,0.69,overlay_alpha),
        'PANLOBULAREMPHYSEMA':(0.00,0.00,0.70,overlay_alpha),
        'VISCERALFAT':(0.58,0.65,0.20,overlay_alpha),
        'INTERMEDIATEBRONCHUS':(0.85,0.75,0.85,overlay_alpha),
        'LOWERLOBEBRONCHUS':(1.00,0.02,0.00,overlay_alpha),
        'SUPERIORDIVISIONBRONCHUS':(0.98,0.50,0.45,overlay_alpha),
        'LINGULARBRONCHUS':(0.00,0.03,1.00,overlay_alpha),
        'MIDDLELOBEBRONCHUS':(0.25,0.88,0.82,overlay_alpha),
        'BRONCHIECTATICAIRWAY':(0.25,0.88,0.81,overlay_alpha),
        'NONBRONCHIECTATICAIRWAY':(0.25,0.87,0.81,overlay_alpha),
        'AMBIGUOUSBRONCHIECTATICAIRWAY':(0.25,0.86,0.81,overlay_alpha),
        'MUSCLE':(0.90,0.00,0.00,overlay_alpha),
        'HERNIA':(0.91,0.00,0.00,overlay_alpha),
        'BONEMARROW':(0.92,0.00,0.00,overlay_alpha),
        'BONE':(0.93,0.00,0.00,overlay_alpha),
        'INTERSTITIALLUNGDISEASE':(0.94,0.00,0.00,overlay_alpha),
        'SUBPLEURALLINE':(0.01,0.02,0.01,overlay_alpha),
        'NODULE':(0.01,0.02,0.02,overlay_alpha),
        'BENIGNNODULE':(0.01,0.02,0.03,overlay_alpha),
        'MALIGNANTNODULE':(0.01,0.02,0.04,overlay_alpha),
        'SEPTUM':(0.01,0.03,0.04,overlay_alpha),
        'FIBRONODULAR':(0.02,0.03,0.04,overlay_alpha),
        'MESOTHELIOMA':(0.03,0.03,0.04,overlay_alpha),
        'NORMALINFLAMED':(0.04,0.03,0.04,overlay_alpha),
        'NORMALNOTINFLAMED':(0.05,0.03,0.04,overlay_alpha),
        'SYSTOLE':(0.05,0.03,0.05,overlay_alpha),
        'DIASTOLE':(0.05,0.03,0.06,overlay_alpha),
        }                  
        
        QcColorConventionsManager.qc_region_type_color_mapping = {
        ('LEFT','PECTORALISMINOR'):(50.0/255.0,200.0/255.0,220.00/255.0,overlay_alpha),
        ('LEFT','PECTORALISMAJOR'):(200.0/255.0,0.00,0.00,overlay_alpha),
        ('LEFT','SUBCUTANEOUSFAT'):(230.0/255.0,220.0/255.0,0.0/255.0,overlay_alpha),
 
        ('RIGHT','PECTORALISMINOR'):(0.0/255,50.00/255.0,250.00/255, overlay_alpha),
        ('RIGHT','PECTORALISMAJOR'):(200.0/255.0,0.0/255.0,200.00/255.0,overlay_alpha),
        ('RIGHT','SUBCUTANEOUSFAT'):(100.0/255.0,80.0/255.0, 50.0/255.0, overlay_alpha), 
        
        ('PARAVERTEBRAL','SUBCUTANEOUSFAT'):(200.0/255.0,0.0/255.0, 0.0/255.0, overlay_alpha), 
        #('ABDOMEN','VISCERALFAT'):(225.0/255.0,225.0/255.0, 10.0/255.0, overlay_alpha), 
        ('ABDOMEN','VISCERALFAT'):(235.0/255.0,235.0/255.0, 10.0/255.0, overlay_alpha), 

        ('PARAVERTEBRAL','MUSCLE'):(0.0/255.0,200.0/255.0, 200.0/255.0, overlay_alpha), 
                                     
        }
        
        color_bound_list = [[QcColorConventionsManager.qc_background, 0 ]] #, [qc_allregionstypes, 1 ]
        
        """ append label value with region and type to color bounds list. (for now, either region or type)"""
        mychestConvenstion =ChestConventions()
        for region in QcColorConventionsManager.qc_color_mapping.keys():
            """ append the color to the color list"""
            """ get the label bounds and append to bounds list. The bounds are lblval:lblval+1"""
            reg_val =  mychestConvenstion.GetChestRegionValueFromName(region)  
            regtypeval = mychestConvenstion.GetValueFromChestRegionAndType(reg_val, 0) 
            color_bound_list.append([QcColorConventionsManager.qc_color_mapping[region], regtypeval])        

        for the_type in QcColorConventionsManager.qc_type_color_mapping.keys():
            """ append the color to the color list"""
            """ get the label bounds and append to bounds list. The bounds are lblval:lblval+1"""
            type_val =  mychestConvenstion.GetChestTypeValueFromName(the_type)   
            regtypeval = mychestConvenstion.GetValueFromChestRegionAndType(0, type_val) 
            color_bound_list.append([QcColorConventionsManager.qc_type_color_mapping[the_type], regtypeval])  

        # hmmm, why is this messing with the region only/type only colors?
        #""" This is not the ideal way to have region/type colors"""
        for region in QcColorConventionsManager.qc_color_mapping.keys():
            reg_val =  mychestConvenstion.GetChestRegionValueFromName(region)
            for the_type in QcColorConventionsManager.qc_type_color_mapping.keys():
                type_val =  mychestConvenstion.GetChestTypeValueFromName(the_type) 
                regtypeval = mychestConvenstion.GetValueFromChestRegionAndType(reg_val, type_val)
                if(reg_val==0 or type_val==0 or regtypeval==0):
                    print("regval="+str(reg_val)+"type_val="+str(type_val)+"regtypeval="+str(regtypeval))
                    print(the_type)
                    print(region)
                if((region,the_type) in QcColorConventionsManager.qc_region_type_color_mapping.keys()):
                    #import pdb
                    #pdb.set_trace()
                    the_colortemp= np.array(QcColorConventionsManager.qc_region_type_color_mapping[(region,the_type)])  
                else:
                    the_colortemp= (np.array(QcColorConventionsManager.qc_color_mapping[region])+np.array(QcColorConventionsManager.qc_type_color_mapping[the_type]))/2
                the_color = np.ndarray.tolist(the_colortemp)
                color_bound_list.append([the_color, regtypeval])  
                                                   
        """ Sort the tuples and turn back to list """
        color_bound_list.sort(key=lambda tup: tup[1]) 
        color_bound_list = list(zip(*color_bound_list))
        color_bound_list[0]=list(color_bound_list[0])
        color_bound_list[1]=list(color_bound_list[1])
        """ Go through list sequentially and add bounds if non-existent"""
        for i,x in enumerate(color_bound_list[1][:-1]):
            if ((color_bound_list[1][i+1] != color_bound_list[1][i]+1)):# and (color_bound_list[0][i] != QcColorConventionsManager.qc_background)):
                color_bound_list[0].insert(i+1, qc_allregionstypes)#QcColorConventionsManager.qc_background)
                color_bound_list[1].insert(i+1, color_bound_list[1][i]+1)
        color_bound_list[0].append(qc_allregionstypes)#QcColorConventionsManager.qc_background)
        color_bound_list[1].append(color_bound_list[1][-1]+1)              
        color_bound_list[0].append(qc_allregionstypes)#QcColorConventionsManager.qc_background)
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

    def get_labelmap_projection(self, in_lm, regions, types, axis='axial'):
        
        """ project along axis"""   
        if (axis=='sagittal'):  
            axis_val=0
        if (axis=='coronal'):  
            axis_val=1              
        if (axis=='axial'):  
            axis_val=2    
        parser = RegionTypeParser(in_lm)

        mychestConvenstion =ChestConventions()
        mask = np.zeros_like(in_lm)
        if(regions):
            if mychestConvenstion.GetChestWildCardName() in regions:
                #extract a labelmap with only chest types
                mask = mychestConvenstion.GetChestRegionFromValue(in_lm)
            else:
                for i in range(0,np.shape(regions)[0]):             
                    reg_val =  mychestConvenstion.GetChestRegionValueFromName(regions[i])
                    mask = np.logical_or(mask, parser.get_mask(chest_region=reg_val) ) 
        if(types):
            if mychestConvenstion.GetChestWildCardName() in types:
                #extract a labelmap with only chest types
                mask = mychestConvenstion.GetChestTypeFromValue(in_lm)
            else:
                for i in range(0,np.shape(types)[0]):   
                    type_val =  mychestConvenstion.GetChestTypeValueFromName(types[i])
                    print(type_val)          
                    mask = np.logical_or(mask, parser.get_mask(chest_type=type_val) )  
        lm_masked = np.copy(in_lm)
        lm_masked[mask==0]=0 
        
        """ if types=None then extract only regions, if regions=None then extract only types.
        Else leave as is, the label values are a combination of regions and types."""
        if(regions is None):
            """ extract only types"""
            lm_masked = mychestConvenstion.GetChestTypeFromValue(lm_masked)
            lm_masked = mychestConvenstion.GetValueFromChestRegionAndType(np.zeros_like(lm_masked), lm_masked)
        if(types is None):
            lm_masked = mychestConvenstion.GetChestRegionFromValue(lm_masked)
            lm_masked = mychestConvenstion.GetValueFromChestRegionAndType(lm_masked, np.zeros_like(lm_masked))
        """  maximum intensity projection  """               
        projection = np.max(lm_masked, axis=axis_val)
        projection = self.rotate_labelmap_images_for_display(projection, axis)
        return([projection])         

                                                      
    def get_ct_overlay(self, in_ct1, in_ct2, in_mask, num_overlays, axis='axial', is_checkerboard = False):
   
        xmax, ymax,zmax = np.max(np.where(in_mask>0), 1)
        xmin, ymin,zmin = np.min(np.where(in_mask>0), 1)

        xmaxct, ymaxct,zmaxct = np.max(np.where(in_ct1>0), 1)
        xminct, yminct,zminct = np.min(np.where(in_ct1>0), 1)        

        bounds_lm = [xmin,xmax, ymin,ymax, zmin,zmax] 
        bounds_ct = [xminct, xmaxct, yminct,ymaxct, zminct,zmaxct ]
        
        bounds_list = self.get_overlay_bounds_from_volume_bounds(bounds_ct, bounds_lm, num_overlays, axis=axis)                                    

        #""" **** for testing purposes"""
        #in_ct1[in_mask==0]=np.min(in_ct1)
        #in_ct2[in_mask==0]=np.min(in_ct2)
        
        """ Exract the list of overlays """
        list_of_cts = []
        import pdb
        #pdb.set_trace()
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
        
                                             
    def get_segmentation_overlay(self, in_ct, in_lm, num_overlays, regions, types, region_type_pairs=None, axis='axial'):

        parser = RegionTypeParser(in_lm)
        mychestConvenstion =ChestConventions()       
        
        """ TODO: Add wildcard"""
        mask = np.zeros_like(in_lm)

        if(region_type_pairs):
            for i in range(0,np.shape(region_type_pairs)[0]): 
                reg_val =  mychestConvenstion.GetChestRegionValueFromName(region_type_pairs[i][0])
                type_val =  mychestConvenstion.GetChestTypeValueFromName(region_type_pairs[i][1]) 
                mask = np.logical_or(mask, parser.get_mask(chest_region=reg_val, chest_type=type_val) )
        if(regions):
            if mychestConvenstion.GetChestWildCardName() in regions:
                #extract a labelmap with only chest types
                mask = mychestConvenstion.GetChestRegionFromValue(in_lm)
            else:
                for i in range(0,np.shape(regions)[0]):             
                    reg_val =  mychestConvenstion.GetChestRegionValueFromName(regions[i])
                    mask = np.logical_or(mask, parser.get_mask(chest_region=reg_val) ) 

        if(types):
            if mychestConvenstion.GetChestWildCardName() in types:
                """ extract a labelmap with only chest regions""" 
                mask = mychestConvenstion.GetChestTypeFromValue(in_lm)
            else:
                for i in range(0,np.shape(types)[0]):   
                    type_val =  mychestConvenstion.GetChestTypeValueFromName(types[i])
                    mask = np.logical_or(mask, parser.get_mask(chest_type=type_val) )                 
        lm_masked = np.copy(in_lm)
        lm_masked[mask==0]=0 
        
        """ if types=None then extract only regions, if regions=None then extract only types.
        Else leave as is... Need to have color mapping for regions and types combined"""
        if((region_type_pairs is None) and (regions is None)):
            """ extract only types"""
            lm_masked = mychestConvenstion.GetChestTypeFromValue(lm_masked)
            lm_masked = mychestConvenstion.GetValueFromChestRegionAndType(np.zeros_like(lm_masked), lm_masked)
        if((region_type_pairs is None) and (types is None)):
            lm_masked = mychestConvenstion.GetChestRegionFromValue(lm_masked)
            
        xmax, ymax,zmax = np.max(np.where(lm_masked>0), 1)
        xmin, ymin,zmin = np.min(np.where(lm_masked>0), 1)

        xmaxct, ymaxct,zmaxct = np.max(np.where(in_ct>0), 1)
        xminct, yminct,zminct = np.min(np.where(in_ct>0), 1)        

        bounds_lm = [xmin,xmax, ymin,ymax, zmin,zmax] 
        bounds_ct = [xminct, xmaxct, yminct,ymaxct, zminct,zmaxct ]
    
        bounds_list = self.get_overlay_bounds_from_volume_bounds(bounds_ct, bounds_lm, num_overlays, axis)                                    

        #pdb.set_trace()
        """ Exract the list of overlays """
        list_of_cts = []
        list_of_labelmaps = []

        for i in range(0,num_overlays):
            ct_patch = np.squeeze(in_ct[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
                bounds_list[i][4]:bounds_list[i][5]]).astype(float)
            
            lm_patch = np.squeeze(lm_masked[bounds_list[i][0]:bounds_list[i][1],bounds_list[i][2]:bounds_list[i][3],\
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
        window_width=1100, window_level=-1024, resolution=50, list_of_voxel_spacing=None):
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
        for i in range(0, num_rows):
            num_columns=len(list_of_cts[i])
            for j in range(0, num_columns):
                ax_1 = fig.add_subplot(gs[i:(i+1),j:(j+1)])

                ax_1.set_xticks([])
                ax_1.set_yticks([])
                ax_1.autoscale_view('tight')
                fig.add_subplot(ax_1)
                
                num_ct_overlays = len(list_of_cts[i][j])
                
                for k in range(0, num_ct_overlays):
                    list_of_cts[i][j][k][list_of_cts[i][j][k]<(int(window_level)-int(window_width)/2)] = (int(window_level)-int(window_width)/2)
                    list_of_cts[i][j][k][list_of_cts[i][j][k]>(int(window_level)+int(window_width)/2)] = (int(window_level)+int(window_width)/2)
                    list_of_cts[i][j][k] = list_of_cts[i][j][k]-np.min(list_of_cts[i][j][k])
                    list_of_cts[i][j][k] = list_of_cts[i][j][k]/float(np.max(list_of_cts[i][j][k])+0.1)  
                       
                im_shape = np.shape(list_of_cts[i][j][0])        
                if(list_of_voxel_spacing is None):
                    the_extent=None
                else:
                    the_extent = (0, list_of_voxel_spacing[i][j][1]*im_shape[1], list_of_voxel_spacing[i][j][0]*im_shape[0] ,0)                         
                if (num_ct_overlays > 1):
                        from PIL import Image
                        im1 = Image.new("RGB", [im_shape[1],im_shape[0]], "black")
                        im2 = Image.new("RGB", [im_shape[1],im_shape[0]], "black")
                        import pdb
                        #pdb.set_trace()
                        #im1 = Image.fromarray(np.uint8(cm.Blues_r(list_of_cts[i][j][1])*255))
                        #im2 = Image.fromarray(np.uint8(cm.YlOrBr_r(list_of_cts[i][j][1])*255))

                        im1_array = np.array(im1)
                        im2_array = np.array(im2) 
                        #im1_array[:,:,0] = (list_of_cts[i][j][0]*255)
                        #import pdb
                        #pdb.set_trace()
                        ct1_prop=(list_of_cts[i][j][0])/(list_of_cts[i][j][0]+list_of_cts[i][j][1])
                        ct2_prop=(list_of_cts[i][j][1])/(list_of_cts[i][j][0]+list_of_cts[i][j][1])
                        
                        im1_array[:,:,0] = (list_of_cts[i][j][0]*43.0)#*ct1_prop
                        im1_array[:,:,1] = (list_of_cts[i][j][0]*80)#*ct1_prop
                        im1_array[:,:,2] = (list_of_cts[i][j][0]*200)#*ct1_prop
##
                        im2_array[:,:,0] = (list_of_cts[i][j][1]*212.0)#*ct2_prop
                        im2_array[:,:,1] = (list_of_cts[i][j][1]*175.0)#*ct2_prop
                        im2_array[:,:,2] = (list_of_cts[i][j][1]*55.0)#*ct2_prop


                        #im1_array[:,:,0] = (list_of_cts[i][j][0]*28.0+list_of_cts[i][j][1]*254.0)/2.0# 130#
                        #im1_array[:,:,1] = (list_of_cts[i][j][0]*164+list_of_cts[i][j][1]*247.0)/2.0
                        #im2_array[:,:,2] = (list_of_cts[i][j][0]*252.0+list_of_cts[i][j][1]*55.0)/2.0
                        # the blend should be ct1_intensity/(ct1i+ct2i)*ct1_r_value+ct2...


                        #im1_array[:,:,0] = (ct1_prop*25.0+ct2_prop*230.0)# 130#
                        #im1_array[:,:,1] = (ct1_prop*102+ct2_prop*153.0)
                        #im2_array[:,:,2] = (ct1_prop*210.0+ct2_prop*45.0)                                                                       
                        image_blend=(im1_array+im2_array)
                        #image_blend = Image.blend(Image.fromarray(im1_array,'RGB'),Image.fromarray(im2_array,'RGB'),alpha=overlay_alpha)
                        im = plt.imshow(np.array(image_blend),origin='upper', extent = the_extent, vmin=0, vmax=0.8) 
                else:
                      myccmap= 'gray' 
                      the_alpha = 1.0
                      im = plt.imshow(list_of_cts[i][j][0],origin='upper', \
                        cmap=plt.get_cmap(myccmap),  clim=(0.0, 1.0), extent = the_extent) #0.2,0.8 was too much #vmin=0.2, vmax=0.8)
                     
        
                if((len(list_of_labelmaps) >0) and (len(list_of_labelmaps[i]) >0) and (len(list_of_labelmaps[i][j]) >0)):
                    """ append the overlay """

                    im=plt.imshow(list_of_labelmaps[i][j][0],origin='upper', \
                        cmap=QcColorConventionsManager.cmap , norm=QcColorConventionsManager.norm,\
                        vmin=1, vmax=100000, extent = the_extent) #norm=norm, alpha=0.95,

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
        
    
    def get_labelmap_qc(self, in_ct, out_file, list_of_labelmaps, list_request_qc_per_labelmap, 
            list_of_axes, num_images_per_region=3, overlay_alpha=0.85, window_width=1100, 
            window_level=-1024, resolution=600,  spacing=None, is_overlay=False, is_projection=False):   

    
        """ the layout of the images should be set here"""
        append_horizontally = False
            
        list_of_images = []
        list_cts = [] 
        list_labelmaps = []
        list_of_voxel_spacing=[]
        print("getting labelmap QC")
        print(list_request_qc_per_labelmap)

        for j in range (0, len(list_of_labelmaps)):
            for i in range(0, len(list_request_qc_per_labelmap[j])):
                
                if self.region_dictionary[list_request_qc_per_labelmap[j][i]]:
                    in_regions= self.region_dictionary[list_request_qc_per_labelmap[j][i]]
                else:
                    in_regions = None
                if self.type_dictionary[list_request_qc_per_labelmap[j][i]]:
                    in_types = self.type_dictionary[list_request_qc_per_labelmap[j][i]]
                else:
                    in_types = None
                if self.region_type_dictionary[list_request_qc_per_labelmap[j][i]]:
                    in_regions_types = self.region_type_dictionary[list_request_qc_per_labelmap[j][i]]
                else:
                    in_regions_types = None
                                        
                overlay = ImageOverlay()
                if ((is_overlay) and (list_request_qc_per_labelmap[j][i] in self.overlay_qc)):
                                    
                    print("generating overlay for the following regions: ")
                    print(in_regions)
                    print(in_types)
  
                    """ get overlay QC"""
                    temp_list_cts,temp_list_labelmaps = overlay.get_segmentation_overlay( in_ct, list_of_labelmaps[j],\
                        num_images_per_region, in_regions, in_types, region_type_pairs=in_regions_types, axis=list_of_axes[j][i])

                    list_cts.append(temp_list_cts)
                    list_labelmaps.append(temp_list_labelmaps)
                     
                    if(spacing):    
                        if (list_of_axes[j][i]=='axial'): 
                            list_of_voxel_spacing.append([[spacing[1],spacing[0]]]*num_images_per_region) 
                        elif(list_of_axes[j][i]=='sagittal'): 
                            list_of_voxel_spacing.append([[spacing[2],spacing[1]]]*num_images_per_region) 
                        elif(list_of_axes[j][i]=='coronal'): 
                            list_of_voxel_spacing.append([[spacing[2],spacing[0]]]*num_images_per_region)                 
                
                #""" get projection QC """
                if((is_projection) and (list_request_qc_per_labelmap[j][i] in self.projection_qc)):

                    print("generating projections for: ")
                    print(in_regions)
                    print(in_types)
                                        
                    overlay_projection = overlay.get_labelmap_projection(list_of_labelmaps[j], in_regions, in_types, axis=list_of_axes[j][i])
                    #temp_list_cts,temp_list_labelmaps = overlay.get_segmentation_overlay( in_ct, list_of_labelmaps[j],\
                    #    1, in_regions, in_types, axis=list_of_axes[j][i])
                        

                    
                    if(spacing):    
                        if (list_of_axes[j][i]=='axial'): 
                            list_of_voxel_spacing.append([[spacing[1],spacing[0]]]*2) 
                        elif(list_of_axes[j][i]=='sagittal'): 
                            list_of_voxel_spacing.append([[spacing[2],spacing[1]]]*2) 
                        elif(list_of_axes[j][i]=='coronal'): 
                            list_of_voxel_spacing.append([[spacing[2],spacing[0]]]*2) 
                    list_cts.append([np.zeros_like(overlay_projection)])
                    list_labelmaps.append([overlay_projection])
                    
        return [list_cts, list_labelmaps, list_of_voxel_spacing]

       
