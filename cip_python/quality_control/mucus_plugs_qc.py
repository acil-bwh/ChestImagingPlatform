# -*- coding: utf-8 -*-
import numpy as np
import argparse
import cip_python.common as common
from cip_python.input_output import ImageReaderWriter
from cip_python.dcnn.data import DataProcessing
import os
import SimpleITK as sitk

from typing import List, Any, Union

class MucusPlugsQC:
    """General purpose class for generating QC images from mucus plugs segmentations or
    xml file with points

    The class extracts each Nodule point from the XML string representation of a
    GeometryTopologyData object and, if avaialble, the corresponding nodule segmentation and
    generations a 2.5D image composition around the point.
       
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
    def __init__(self, x_extent=32, y_extent=32, z_extent=32):
        self.x_extent = int(x_extent)
        self.y_extent = int(y_extent)
        self.z_extent = int(z_extent)
        self.window_width = 1000
        self.window_level = -500
        self.sampling_spacing = 0.625  #1 mm
        self.output_im_dim=np.array([256,256])
        self.alpha = 0.5

    def resample_im(self,im_sitk,spacing,labelmap=False):

        if labelmap==True:
            interpolator_type=sitk.sitkNearestNeighbor
            output_type=sitk.sitkUInt16
        else:
            interpolator_type=sitk.sitkBSpline
            output_type=sitk.sitkInt16

        oo=DataProcessing()

        return oo.resample_image_itk_by_spacing(im_sitk,[spacing,spacing,spacing],output_type,interpolator_type)

    def execute(self,ct_sitk,mp_lm,xml_data=None):
        """ Execute QC image extraction

        Parameters
        ----------
        ct_sitk: SimpleITK image object
            CT Image
        xml_data: String
            XML string representation of a  GeometryTopologyData object
        mp_lm: SimpleITK image object (optional)
            Label map containing nodule information to display as overlay
        """

        #Resample images to common space and pad to account for extent
        ct_resample=self.resample_im(ct_sitk,self.sampling_spacing,labelmap=False)
        pad_ff = sitk.ConstantPadImageFilter()
        pad_ff.SetPadLowerBound([int(self.x_extent / 2), int(self.y_extent / 2), int(self.z_extent / 2)])
        pad_ff.SetPadUpperBound([int(self.x_extent / 2), int(self.y_extent / 2), int(self.z_extent / 2)])
        ct_pad = pad_ff.Execute(ct_resample)

        if mp_lm is not None:
            mp_resample=self.resample_im(mp_lm,self.sampling_spacing,labelmap=True)
        else:
            mp_resample=None

        centroids = {}

        if xml_data is not None:
            #Get points to display from xml
            my_geometry_data = common.GeometryTopologyData.from_xml(xml_data)

            # loop through each point and create a patch around it
            myChestConventions = common.ChestConventions()
            for label,the_point in zip(range(len(my_geometry_data.points)),my_geometry_data.points):
                #if the_point.chest_type == 86 or the_point.chest_type == 87 or the_point.chest_type == 88:
                pp_xyz = the_point.coordinate

                centroids[label]=pp_xyz
        else:

            connected_components = sitk.ConnectedComponent(mp_lm)

            label_stats = sitk.LabelShapeStatisticsImageFilter()
            label_stats.Execute(connected_components)

            components = [
                {
                    "Label": label,
                    "Size": label_stats.GetPhysicalSize(label),
                    "Pixels": label_stats.GetNumberOfPixels(label),
                    "Centroid": label_stats.GetCentroid(label)
                }
                for label in label_stats.GetLabels()
            ]

            # Step 4: Sort by size (largest to smallest)
            components_sorted = sorted(components, key=lambda x: x["Size"], reverse=True)

            # Step 5: Record centroids and size in sorted order
            sorted_centroids = [component["Centroid"] for component in components_sorted]
            sorted_pixels = [component["Pixels"] for component in components_sorted]

            for ll in range(len(sorted_centroids)):
                centroids[ll]=sorted_centroids[ll]
                print("label {}: size (pixels)={}, coord={}".format(ll,sorted_pixels[ll],centroids[ll]))


        montage_im_list=list()

        for label, pp_xyz in centroids.items():

            #1. Extract close-up views around nodule

            # Transform coordinates to ijk from pad image
            pp_ijk=ct_pad.TransformPhysicalPointToIndex(pp_xyz)

            #Define region of the image block to Extract
            region_size=[self.x_extent,self.y_extent,self.z_extent]
            region_or=[int(pp_ijk[0]-self.x_extent/2.0),int(pp_ijk[1]-self.y_extent/2.0),int(pp_ijk[2]-self.z_extent/2.0)]

            ct_block=sitk.RegionOfInterest(ct_pad,region_size,region_or)

            #ct_block = ee.Execute(ct_resample)
            if mp_lm is not None:
                mp_block=sitk.RegionOfInterest(pad_ff.Execute(mp_resample), region_size, region_or)
            else:
                mp_block=None

            qc_montage_block=self.create_25d_montage(ct_block,mp_block)

            #2. Extract full slice views
            #Get ijk coordinate in the resample image space
            pp_ijk=ct_resample.TransformPhysicalPointToIndex(pp_xyz)
            im_25d_full, _tmp = self.extract_25d(ct_resample, pp_ijk)

            #Add red dot
            #Define factor
            or_size=ct_resample.GetSize()

            for plane in ['ax','cor']:
                upper_corner = []
                bottom_corner = []
                if plane=='ax':
                    im_size=im_25d_full[plane].GetSize()
                    factor_i=im_size[0]/or_size[0]
                    factor_j=im_size[1]/or_size[1]
                    i_coord=int(im_size[0]/or_size[0] * pp_ijk[0])
                    j_coord=int(im_size[1]/or_size[1] * pp_ijk[1])
                    upper_corner.append(int(factor_i*(pp_ijk[0]-self.x_extent/2.0)))
                    upper_corner.append(int(factor_j*(pp_ijk[1]-self.y_extent/2.0)))
                    bottom_corner.append(int(factor_i*(pp_ijk[0]+self.x_extent/2.0)))
                    bottom_corner.append(int(factor_j*(pp_ijk[1]+self.y_extent/2.0)))
                    flip_ax=None
                elif plane=='sag':
                    im_size = im_25d_full[plane].GetSize()
                    factor_i=im_size[0] / or_size[1]
                    factor_j=im_size[1] / or_size[2]
                    i_coord = int(factor_i * pp_ijk[1])
                    j_coord = int(factor_j * pp_ijk[2])
                    flip_ax=[True,False]
                elif plane=='cor':
                    im_size = im_25d_full[plane].GetSize()
                    factor_i=im_size[0] / or_size[0]
                    factor_j=im_size[1] / or_size[2]
                    i_coord = int(im_size[0] / or_size[0] * pp_ijk[0])
                    j_coord = int(im_size[1] / or_size[2] * pp_ijk[2])
                    bottom_corner.append(int(factor_i*(pp_ijk[0]+self.x_extent/2.0)))
                    bottom_corner.append(int(factor_j*(pp_ijk[2]+self.y_extent/2.0)))
                    upper_corner.append(int(factor_i*(pp_ijk[0]-self.x_extent/2.0)))
                    upper_corner.append(int(factor_j*(pp_ijk[2]-self.y_extent/2.0)))
                    flip_ax=[False,True]

                #Draw point
                self.draw_point(im_25d_full[plane],[i_coord,j_coord])
                #Draw rectanble
                im_25d_full[plane]=self.draw_rectangle(im_25d_full[plane],upper_corner,bottom_corner)

                #Flip image for display
                if flip_ax is not None:
                    flip_ff=sitk.FlipImageFilter()
                    flip_ff.SetFlipAxes(flip_ax)
                    im_25d_full[plane]=flip_ff.Execute(im_25d_full[plane])

            #Final Tiling
            if mp_lm is not None:
                tile_order=(1,2)
            else:
                tile_order=(2,1)
            qc_montage_full=sitk.Tile([im_25d_full['ax'],im_25d_full['cor']],tile_order)
            qc_montage=sitk.Tile([qc_montage_full,qc_montage_block],(2,1))

            montage_im_list.append(qc_montage)

        return montage_im_list

    def draw_point(self,image,center):
        i_coord=center[0]
        j_coord=center[1]
        image[i_coord, j_coord] = [255, 0, 0]
        image[i_coord + 1, j_coord] = [255, 0, 0]
        image[i_coord, j_coord + 1] = [255, 0, 0]
        image[i_coord + 1, j_coord + 1] = [255, 0, 0]
        image[i_coord - 1, j_coord] = [255, 0, 0]
        image[i_coord, j_coord - 1] = [255, 0, 0]
        image[i_coord - 1, j_coord - 1] = [255, 0, 0]

    def draw_rectangle(self,image,upper_corner,lower_corner):
        #check boundaries
        pp1=[]
        pp2=[]
        im_size=image.GetSize()
        for cc,ss, in zip(upper_corner,im_size):
            if cc<0:
                cc=0
            if cc>ss:
                cc=ss-1
            pp1.append(cc)
        for cc,ss, in zip(lower_corner,im_size):
            if cc<0:
                cc=0
            if cc>ss:
                cc=ss-1
            pp2.append(cc)

        for pp in range(pp1[0],pp2[0]):
            image[pp,pp1[1]]=[255,0,0]
            image[pp,pp2[1]]=[255,0,0]
        for pp in range(pp1[1],pp2[1]):
            image[pp1[0],pp]=[255,0,0]
            image[pp2[0],pp]=[255,0,0]
        return image


    def save_montage(self,montage_im_list,file_prefix):

        for id_label,qc_im in enumerate(montage_im_list):
            #Create file name
            file_name=os.path.join("{}_p{:0>3}_mucusQCMontage.png".format(file_prefix,id_label))
            sitk.WriteImage(qc_im,file_name)
            #qc_np=sitk.GetArrayFromImage(qc_im)
            #scipy.misc.toimage(qc_np).save(file_name)

    def display_im(self,im):
        image_viewer = sitk.ImageViewer()
        image_viewer.SetTitle('grid using ImageViewer class')
        # Use the default image viewer.
        image_viewer.SetApplication('/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP')
        image_viewer.Execute(im)

    def create_25d_montage(self,image,labelmap=None):

        im_size=image.GetSize()
        center=list()
        for val in im_size:
            center.append(np.int16(np.round(val/2.0)))
        im_25d,lm_25d=self.extract_25d(image,center,labelmap,color=sitk.ScalarToRGBColormapImageFilter.Red,alpha=self.alpha)

        #Arrange output in montage
        all_montage=list()
        for plane in ['ax','sag','cor']:
            all_montage.append(im_25d[plane])
        if len(lm_25d):
            for plane in ['ax','sag','cor']:
                    all_montage.append(lm_25d[plane])
        output_qc_im=sitk.Tile(all_montage,(3,2))

        #out_np=np.concatenate((sitk.GetArrayFromImage(im_25d['ax']),sitk.GetArrayFromImage(im_25d['sag']),sitk.GetArrayFromImage(im_25d['cor'])),axis=1)
        #if len(lm_25d)>0:
        #    panel2=np.concatenate((sitk.GetArrayFromImage(im_25d['ax']),sitk.GetArrayFromImage(im_25d['sag']),sitk.GetArrayFromImage(im_25d['cor'])),axis=1)
        #    out_np=np.concatenate((out_np,panel2),axis=0)
        #output_qc_im=sitk.GetImageFromArray(out_np,isVector=True)
        #output_qc_im.SetSpacing((1.,1.,1.))

        return output_qc_im

    def extract_25d(self,im,center,lm=None,color=sitk.ScalarToRGBColormapImageFilter.Red,alpha=0.1):

        im_25d=dict()
        or_25d=dict()
        lm_25d=dict()
        image_mappings = {'grey': lambda x: sitk.Compose([x] * 3),
                          'jet': lambda x: sitk.ScalarToRGBColormap(x, sitk.ScalarToRGBColormapImageFilter.Jet),
                          'hot': lambda x: sitk.ScalarToRGBColormap(x, sitk.ScalarToRGBColormapImageFilter.Hot)}

        wl_f=sitk.IntensityWindowingImageFilter()

        # Transform gray-scale
        wl_f.SetWindowMinimum(self.window_level-self.window_width//2)
        wl_f.SetWindowMaximum(self.window_level+self.window_width//2)
        #wl_f.SetWindowMinimum(-900)
        #wl_f.SetWindowMaximum(-400)
        wl_f.SetOutputMinimum(0)
        wl_f.SetOutputMaximum(255)


        for plane in ['ax','sag','cor']:
            im_25d[plane]=image_mappings['grey'](sitk.Cast(wl_f.Execute(self.extract_plane(im,center,plane)),sitk.sitkUInt8))
            or_25d[plane]=wl_f.Execute(self.extract_plane(im,center,plane))
        if lm is not None:
            #Alpha blending
            for plane in ['ax','sag','cor']:
                lm_rgb = sitk.ScalarToRGBColormap(sitk.Cast(255*self.extract_plane(lm,center,plane,True),sitk.sitkUInt8), color)
                lm_25d[plane]=sitk.Cast(self.alpha_blend(image1=im_25d[plane],image2=lm_rgb ,alpha=alpha, mask2= self.extract_plane(lm,center,plane)>=1 ), sitk.sitkVectorUInt8)

                #alpha_im = sitk.Image(lm_rgb.GetSize(), sitk.sitkFloat32) + alpha
                #alpha_im.CopyInformation(lm_rgb)
                #lm_25d[plane]=sitk.Cast( alpha_im*sitk.Cast(im_25d[plane],sitk.sitkVectorFloat32)+(1-alpha_im)*sitk.Cast(lm_rgb,sitk.sitkVectorFloat32),sitk.sitkVectorUInt8)

                #lm_rgb = 255*self.extract_plane(lm,center,plane,True)
                #lm_25d[plane]=sitk.LabelOverlay(image=or_25d[plane],labelImage=lm_rgb,opacity=0.5,backgroundValue=0,colormap=[255,0,0])
        return im_25d,lm_25d

    def extract_plane(self,im,center,plane,labelmap=False):

        if labelmap==True:
            interpolator=sitk.sitkLinear
        else:
            interpolator=sitk.sitkNearestNeighbor

        oo=DataProcessing()
        if plane=='ax':
            return oo.resample_image_itk(im[:,:,int(center[2])],self.output_im_dim,interpolator=interpolator)[0]
        if plane=='sag':
            return oo.resample_image_itk(im[int(center[0]),:,:],self.output_im_dim,interpolator=interpolator)[0]
        if plane=='cor':
            return oo.resample_image_itk(im[:,int(center[1]),:],self.output_im_dim,interpolator=interpolator)[0]


    def alpha_blend(self,image1, image2, alpha=0.5, mask1=None, mask2=None):
        '''
        Alaph blend two images, pixels can be scalars or vectors.
        The alpha blending factor can be either a scalar or an image whose
        pixel type is sitkFloat32 and values are in [0,1].
        The region that is alpha blended is controled by the given masks.
        '''

        if not mask1:
            mask1 = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + 1.0
            mask1.CopyInformation(image1)
        else:
            mask1 = sitk.Cast(mask1, sitk.sitkFloat32)
        if not mask2:
            mask2 = sitk.Image(image2.GetSize(), sitk.sitkFloat32) + 1
            mask2.CopyInformation(image2)
        else:
            mask2 = sitk.Cast(mask2, sitk.sitkFloat32)
        # if we received a scalar, convert it to an image
        if type(alpha) != sitk.SimpleITK.Image:
            alpha = sitk.Image(image1.GetSize(), sitk.sitkFloat32) + alpha
            alpha.CopyInformation(image1)
        components_per_pixel = image1.GetNumberOfComponentsPerPixel()
        if components_per_pixel > 1:
            img1 = sitk.Cast(image1, sitk.sitkVectorFloat32)
            img2 = sitk.Cast(image2, sitk.sitkVectorFloat32)
        else:
            img1 = sitk.Cast(image1, sitk.sitkFloat32)
            img2 = sitk.Cast(image2, sitk.sitkFloat32)

        intersection_mask = mask1 * mask2

        intersection_image = self.mask_image_multiply(alpha * intersection_mask, img1) + \
                             self.mask_image_multiply((1 - alpha) * intersection_mask, img2)

        #return intersection_image

        return intersection_image + self.mask_image_multiply(mask2 - intersection_mask, img2) + \
               self.mask_image_multiply(mask1 - intersection_mask, img1)

    def mask_image_multiply(self,mask, image):
        components_per_pixel = image.GetNumberOfComponentsPerPixel()
        if  components_per_pixel == 1:
            return mask*image
        else:
            return sitk.Compose([mask*sitk.VectorIndexSelectionCast(image,channel) for channel in range(components_per_pixel)])

             
if __name__ == "__main__":
    desc = """ Generate QC image to adjudicate mucus plugs detections and segmentations"""
    
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--in_ct',
                      help='Input ct file name', dest='in_ct', metavar='<string>', required=True)          
    parser.add_argument('--in_lm',
                      help='Input mucus plugs labelmap file name', dest='in_lm', metavar='<string>',
                      default=None)
    parser.add_argument('--in_xml',
                      help='Input xml points file name', dest='in_xml', metavar='<string>',
                      default=None)                                                        
    parser.add_argument('--x_extent',
                      help='x extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='x_extent', 
                      metavar='<int>', type=int, default=64)
    parser.add_argument('--y_extent',
                      help='y extent of each ROI in which the features will be \
                      computed.  (optional)', dest='y_extent',
                      metavar='<int>', type=int, default=64)
    parser.add_argument('--z_extent',
                      help='z extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='z_extent', 
                      metavar='<int>', type=int, default=64)
    parser.add_argument('--output_prefix',
                      help='Prefix used for output QC images', dest='output_prefix', metavar='<string>', required=True,
                      default=None)   
	                                                                            
	                                                                            	                                                                            	                                                                            
    options = parser.parse_args()


    print ("Reading input files for case "+options.in_ct)
    image_io = ImageReaderWriter()
    ct=image_io.read(options.in_ct)
    if options.in_lm is not None:
        mp_lm=image_io.read(options.in_lm)
    else:
        mp_lm=None

    if options.in_xml is not None:
        with open(options.in_xml, 'r+b') as f:
            xml_data = f.read()
    else:
        xml_data=None

    nqc=MucusPlugsQC(x_extent = np.int16(options.x_extent), \
        y_extent=np.int16(options.y_extent), z_extent=np.int16(options.z_extent))

    montage=nqc.execute(ct,mp_lm,xml_data)
    nqc.save_montage(montage,options.output_prefix)
