# -*- coding: utf-8 -*-
import numpy as np
from optparse import OptionParser
import cip_python.common as common
from cip_python.input_output import ImageReaderWriter
from cip_python.classification.get_ct_patch_from_center import Patcher
import vtk
import os   
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
     
               
class xMLPointsQC:
    """General purpose class for generating QC images from an 
    xml file of points. 

    The class extracts each point from the XML string representation of a  
    GeometryTopologyData object and transform it from
    lps to ijk space. Then, a patch of the CT image around the ijk point
    (and an associated labelmap) is defined and extracted and a QC image
    in generated. 
    
    example usage: ~/ChestImagingPlatform-build/CIPPython-install/bin/python 
    xml_points_qc.py --in_ct 11083P_INSP_SHARP_UIA_COPD.nrrd --in_lm
    11083P_INSP_SHARP_UIA_COPD_wholeLung.nrrd --in_xml 
    11083P_INSP_SHARP_UIA_COPD_parenchymaTraining.xml --output_prefix 
    ~/Documents/Data/11083P_INSP_SHARP_UIA_COPD
       
    Parameters 
    ----------    
        
    x_extent: int
        region size in the x direction over which the patch will
        be extracted. The region will be centered at the xml point.
            
    y_extent: int
        region size in the y direction over which the patch will
        be extracted. The region will be centered at the xml point.
        
    z_extent: int
        region size in the z direction over which the patch will
        be extracted. The region will be centered at the xml point.
                               
    """
    def __init__(self, x_extent = 31, y_extent=31, z_extent=1):
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.z_extent = z_extent
        
    
        
    def exctract_case_patches(self, ct, ct_header, lm, xml_object):
        """ Extract CT patches for each case 
        
        Parameters
        ----------        
        ct: 3D numpy array, shape (L, M, N)
            Input CT image from which histogram information will be derived
    
        ct_header: header information for the CT. Should be a dictionary with 
            the following entries: origin, direction, spacing.
            
        lm: 3D numpy array, shape (L, M, N)
            Input mask.  
            
        xml_object: String
            XML string representation of a  GeometryTopologyData object  
            
        """                
        #build transformation matrix
        the_origin = np.array(ct_header['space origin'])
        the_direction = np.reshape(np.array(ct_header['space directions']), [3,3])
        the_spacing = np.array(ct_header['spacing'])

        matrix = np.zeros([4,4])
        matrix[0:3,3] = the_origin
        for i in range(0,3):
            matrix[i,0:3] = the_direction[i,0:3]*the_spacing[i]
        matrix[3,3] = 1
        transformationMatrix=vtk.vtkMatrix4x4()
        transformationMatrix.Identity()
        for i in range(0,4):
            for j in range(0,4):
                transformationMatrix.SetElement(i, j, matrix[i,j])
    
        transformationMatrix.Invert()
        # extract points
        my_geometry_data = common.GeometryTopologyData.from_xml(xml_object)
        
        # loop through each point and create a patch around it
        inc = 0
        myChestConventions = common.ChestConventions()
        case_patch_list = []
        for the_point in my_geometry_data.points : 
            coordinates = the_point.coordinate
            
            # Get coords in ijk space    
            ijk_val = transformationMatrix.MultiplyPoint([coordinates[0],\
                coordinates[1],coordinates[2],1]) # need to append a 1 at th eend of point

            # from here we can build the patches ... 
            test_bounds = Patcher.get_bounds_from_center(ct,ijk_val,[self.x_extent,\
                self.y_extent,self.z_extent])
            ct_patch = Patcher.get_patch_given_bounds(ct,test_bounds)
            lm_patch = Patcher.get_patch_given_bounds(lm,test_bounds)
                  
            chest_type = myChestConventions.GetChestTypeName(the_point.chest_type)

            if(ct_patch != None):    
               case_patch_list.append([ct_patch,lm_patch,chest_type, test_bounds])
            else:
               print("Error! point "+str(inc)+" out of bounds!! "+\
                str(coordinates[0])+" "+str(coordinates[1])+" "+str(coordinates[2])+" mapped to "+\
                    str(ijk_val[0])+" "+str(ijk_val[1])+" "+str(ijk_val[2]))  
               print("bounds are : "+str(test_bounds[0])+" "+str(test_bounds[1])+" "+str(test_bounds[2])+" "+\
                str(test_bounds[3])+" "+str(test_bounds[4])+" "+str(test_bounds[5]))
            inc = inc+1  
        return(case_patch_list)


    def generate_case_images(self, ct, ct_header, lm, xml_object, output_prefix):
        """Generate QC images for each case 
        
        Parameters
        ----------        
        ct: 3D numpy array, shape (L, M, N)
            Input CT image from which histogram information will be derived
    
        ct_header: header information for the CT. Should be a dictionary with 
            the following entries: origin, direction, spacing.
            
        lm: 3D numpy array, shape (L, M, N)
            Input mask.  
            
        xml_object: String
            XML string representation of a  GeometryTopologyData object  
            
        output_prefix: String 
            File prefix for the output images. 
            
        """

        print ("Extracting patches...")
        #intensity_extractor = self.exctract_case_patches( x_extent = np.int16(options.x_extent), \
        
            #y_extent=np.int16(options.y_extent), z_extent=np.int16(options.z_extent))

        patch_list_c = self.exctract_case_patches(ct, ct_header, lm, xml_object)    

        print("generating Qc for case")
        num_points = np.shape(patch_list_c)[0]
        
        for point_ix in range(0,num_points):
            if(np.mod(point_ix,20)==0):
                print("point: "+str(point_ix))
            patch = patch_list_c[point_ix][0]  
                
            bounds = [patch_list_c[point_ix][3][0],patch_list_c[point_ix][3][1],patch_list_c[point_ix][3][2], \
                patch_list_c[point_ix][3][3],patch_list_c[point_ix][3][4],patch_list_c[point_ix][3][5]] 
                
            full_slice = np.squeeze(ct[:,:,bounds[4]:bounds[5]]).astype(float)
                
            x = np.float(patch_list_c[point_ix][3][0])# + np.float(options.x_extent)/2.0
            y = np.float(patch_list_c[point_ix][3][2])# + np.float(options.y_extent)/2.0   
            x_size= np.float(patch_list_c[point_ix][3][1] - patch_list_c[point_ix][3][0])
            y_size= np.float(patch_list_c[point_ix][3][3] - patch_list_c[point_ix][3][2])
                
            # level: min val below which every thing is 0
            # window: width of intensity interval. level + widthis value above which everything is 1
            window_width =  527
            window_level = -993 

            full_slice[full_slice<window_level] = window_level
            full_slice[full_slice>(window_level+window_width)] = (window_level+window_width)
            full_slice = full_slice-np.min(full_slice)
            full_slice = full_slice/float(np.max(full_slice))         
                
            transform=np.array([[0,1],[1,0]])
            full_slice=scipy.ndimage.interpolation.affine_transform(\
                full_slice,transform.T,order=2,cval=0.0,output=np.float32, \
                output_shape=np.shape(np.transpose(full_slice))) 
 
            full_slice2 = (scipy.ndimage.zoom(full_slice, 2, order=2)).astype(float)
            patch[patch<window_level] = window_level
            patch[patch>(window_level+window_width)] = (window_level+window_width)
            patch = patch-np.min(patch)
            patch = patch/float(np.max(patch))           


            # resize here
            # patch2=scipy.misc.imresize(patch, 10, interp='bilinear')
            # patch2=(scipy.ndimage.zoom(patch, 4, order=2)).astype(float)

                
            patch2=patch
            patch2=scipy.ndimage.interpolation.affine_transform(\
                patch2,transform.T,order=2,\
                cval=0.0,output=np.float32, output_shape=np.shape(np.transpose(patch2)))
                            
            print("generating figure")
            fig = plt.figure()  
            print("fig generated")    
            gs=GridSpec(9,10) # 2 rows, 3 columns
                                            
            #ax_1 = fig.add_subplot(gs[0:8,1:9])
            ax_1 = fig.add_subplot(gs[0:9,0:9])
            ax_1.axes.get_xaxis().set_visible(False)
            ax_1.axes.get_yaxis().set_visible(False)

            im = plt.imshow(full_slice,origin='upper', \
                cmap=cm.gray, vmin=0., vmax=1.0)  # create the base map 
            rects = [patches.Rectangle(xy=[x,y], width=x_size, \
                 height=y_size,edgecolor='red', facecolor='none')]
            ax_1.add_artist(rects[0])
            rects[0].set_lw(2)

            ax_2 = fig.add_subplot(gs[8,9])
            ax_2.axes.get_xaxis().set_visible(False)
            ax_2.axes.get_yaxis().set_visible(False)
            im2 = plt.imshow(patch2,origin='upper', cmap = cm.Greys_r)   #,interpolation='linear'
                
                
            # create filename
            the_type = patch_list_c[point_ix][2]  
            out_file = os.path.join(output_prefix + "_"+the_type+"_p"+str(point_ix)+"_slice"+str(bounds[4])+"AxialOverlay.png")
                
            fig1 = plt.gcf()
            fig1.savefig(out_file,bbox_inches='tight', dpi=300)
            fig1.clf() 
            fig.clf()
            plt.clf()

            plt.close()                

                        
             
if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input ct file name', dest='in_ct', metavar='<string>',
                      default=None)          
    parser.add_option('--in_lm',
                      help='Input lung labelmap file name', dest='in_lm', metavar='<string>',
                      default=None)
    parser.add_option('--in_xml',
                      help='Input xml points file name', dest='in_xml', metavar='<string>',
                      default=None)                                                        
    parser.add_option('--x_extent',
                      help='x extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='x_extent', 
                      metavar='<string>', default=31)                        
    parser.add_option('--y_extent',
                      help='y extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='y_extent', 
                      metavar='<string>', default=31)   
    parser.add_option('--z_extent',
                      help='z extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='z_extent', 
                      metavar='<string>', default=1)     
    parser.add_option('--output_prefix',
                      help='Prefix used for output QC images', dest='output_prefix', metavar='<string>',
                      default=None)   
	                                                                            
	                                                                            	                                                                            	                                                                            
    (options, args) = parser.parse_args()
    
                                                                     
    print ("Reading input files for case "+options.in_ct)
    image_io = ImageReaderWriter()
    ct, ct_header=image_io.read_in_numpy(options.in_ct)
    lm, lm_header = image_io.read_in_numpy(options.in_lm) 
 
        
    with open(options.in_xml, 'r+b') as f:
        xml_data = f.read()
            
    my_xMLPointsQC = xMLPointsQC(x_extent = np.int16(options.x_extent), \
        y_extent=np.int16(options.y_extent), z_extent=np.int16(options.z_extent))
    my_xMLPointsQC.generate_case_images(ct, ct_header, lm, xml_data, options.output_prefix)
