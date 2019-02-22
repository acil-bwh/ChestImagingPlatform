import sys
from optparse import OptionParser

import numpy as np
import pandas as pd

from cip_python.classification import kdeHistExtractor
import vtk
import nrrd
import cip_python.common as common
from cip_python.input_output.image_reader_writer import ImageReaderWriter
from cip_python.common import ChestConventions


class kdeHistExtractorFromXML:
    """General purpose class implementing a kernel density estimation histogram
    extraction given points from an xml file. 

    The class extracts each point from the xml object and transforms them from
    lps to ijk space. Then, a patch of the CT image around the ijk point is defined
    and features are extracted from that patch
    
       
    Parameters 
    ----------    
    kde_lower_limit:int
        Lower limit of histogram for the kde
        
    kde_upper_limit:int
        Upper limit of histogram for the kde
        
    x_extent: int
        region size in the x direction over which the kde will
        be estimated. The region will be centered at the xml point.
            
    y_extent: int
        region size in the y direction over which the kde will
        be estimated. The region will be centered at the xml point.
        
    z_extent: int
        region size in the z direction over which the kde will
        be estimated. The region will be centered at the xml point.
 
                             
    Attribues
    ---------
    df_ : Pandas dataframe
        Contains the computed histogram information. The 'patch_label' column
        corresponds to the xml point index. A patch is generated from 
        using the point as center and using the x,y, and z extents. 
        The remaining columns are named 'huNum', where 'Num' ranges 
        from 'lower_limit' to 'upper_limit'. These columns record the number of 
        counts for each Hounsfield unit (hu) estimated by the KDE.        
    """
    def __init__(self, lower_limit=-1050, upper_limit=3050, x_extent = 31, \
        y_extent=31, z_extent=1,  lm=None):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.x_extent = x_extent
        self.y_extent = y_extent
        self.z_extent = z_extent
        
        self.df_ = None

    
        
    def fit(self, ct, ct_header, lm, xml_object, partial_lung_lm=None):
        """Compute the histogram of each patch defined in 'patch_labels' beneath
        non-zero voxels in 'lm' using the CT data in 'ct'.
        
        Parameters
        ----------        
        ct: 3D numpy array, shape (L, M, N)
            Input CT image from which histogram information will be derived
    
        ct_header: header information for the CT. Should be a dictionary with 
            the following entries: origin, direction, spacing.
        lm: 3D numpy array, shape (L, M, N)
            Input mask where histograms will be computed.    
            
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
        inc = 1
        the_patch = np.zeros_like(lm)
        
        mychestConvenstion =ChestConventions()
        regions = dict()
        for the_point in my_geometry_data.points : 
            coordinates = the_point.coordinate
              
            ijk_val = transformationMatrix.MultiplyPoint([coordinates[0],\
                coordinates[1],coordinates[2],1]) # need to append a 1 at th eend of point
            if (partial_lung_lm is None):
                    regions[inc] = 'UndefinedRegion'
            else:
                    regions[inc] = mychestConvenstion.GetChestRegionName(partial_lung_lm[int(ijk_val[0]), \
                int(ijk_val[1]), int(ijk_val[2])])
            # from here we can build the patches ...       
            the_patch[int(ijk_val[0])-2:int(ijk_val[0])+3, int(ijk_val[1])- \
                2:int(ijk_val[1])+3,int(ijk_val[2])] = inc
            inc = inc+1
        
        self.hist_extractor = kdeHistExtractor(lower_limit=self.lower_limit, \
            upper_limit=self.upper_limit, \
            x_extent = self.x_extent, y_extent=self.y_extent, \
            z_extent=self.z_extent)
        self.hist_extractor.fit( ct, lm, the_patch) # df will have region / type entries
        self.df_ = pd.DataFrame(columns=self.hist_extractor.df_.columns)
        self.df_ = self.hist_extractor.df_

        inc = 1
        for the_point in my_geometry_data.points : 
            index = self.df_['patch_label'] == inc          
            self.df_.ix[index, 'ChestRegion'] = regions[inc]
                #mychestConvenstion.GetChestRegionName(the_point.chest_region)
            self.df_.ix[index, 'ChestType'] = \
                mychestConvenstion.GetChestTypeName(the_point.chest_type)
            inc = inc+1

                
if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file', dest='in_ct', metavar='<string>',
                      default=None)
    parser.add_option('--in_lm',
                      help='Input mask file. The histogram will only be \
                      computed in areas where the mask is > 0. If lm is not \
                      included, the histogram will be computed everywhere.', 
                      dest='in_lm', metavar='<string>', default=None)                     
    parser.add_option('--in_xml',
                      help='Input xml File. Contains points around which ROIs \
                      will be created for computing the kde', 
                      dest='in_xml', metavar='<string>', default=None) 
    parser.add_option('--out_csv',
                      help='Output csv file with the features', dest='out_csv', 
                      metavar='<string>', default=None)            
    parser.add_option('--lower_limit',
                      help='lower histogram limit.  (optional)',  dest='lower_limit', 
                      metavar='<string>', default=-1050)                        
    parser.add_option('--upper_limit',
                      help='upper histogram limit.  (optional)',  dest='upper_limit', 
                      metavar='<string>', default=3050)  
    parser.add_option('--x_extent',
                      help='x extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='x_extent', 
                      metavar='<string>', default=30)                        
    parser.add_option('--y_extent',
                      help='y extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='y_extent', 
                      metavar='<string>', default=30)   
    parser.add_option('--z_extent',
                      help='z extent of each ROI in which the features will be \
                      computed.  (optional)',  dest='z_extent', 
                      metavar='<string>', default=1)                         
    (options, args) = parser.parse_args()
    
    print ("Reading CT...")

    image_io = ImageReaderWriter()
    ct, ct_header=image_io.read_in_numpy(options.in_ct)
    if (options.in_lm is not None):
        print ("Reading mask...") 
        lm, lm_header = nrrd.read(options.in_lm) 
    else:
         lm = np.ones(np.shape(ct))   

    with open(options.in_xml, 'r+b') as f:
        xml_data = f.read()
    print ("Compute histogram features...")
    kde_feature_extractor = kdeHistExtractorFromXML(lower_limit=np.int16(options.lower_limit), \
        upper_limit=np.int16(options.upper_limit), x_extent = np.int16(options.x_extent), \
        y_extent=np.int16(options.y_extent), z_extent=np.int16(options.z_extent))

    kde_feature_extractor.fit(ct, ct_header, lm, xml_data)    
    if options.out_csv is not None:
        print ("Writing..."+options.out_csv)
        kde_feature_extractor.df_.to_csv(options.out_csv, index=False)
        
