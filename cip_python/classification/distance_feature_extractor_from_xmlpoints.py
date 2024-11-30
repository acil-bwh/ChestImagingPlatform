from optparse import OptionParser

import nrrd
import numpy as np
import pandas as pd
import vtk

from cip_python.common import GeometryTopologyData
from cip_python.classification import DistanceFeatureExtractor
from cip_python.input_output.image_reader_writer import ImageReaderWriter


class DistanceFeatureExtractorFromXML:
    """General purpose class implementing a distance feature extractor from
    an xml file. 

    The class extracts each point from the xml object and transforms them from
    lps to ijk space. Then, a patch of the CT image around the ijk point is defined
    and features are extracted from that patch
       
    Parameters 
    ----------        
    chest_region : string
        Chest region over which the distance was computed. This will be 
        added to the dataframe as a column.
        
    chest_type : string
        Chest type over which the distance was computed.

    pairs : lists of strings
        Two element list indicating a region-type pair for which the distance
        was computed. The first entry indicates the chest region of the pair, 
        and second entry indicates the chest type of the pair. If more than 1 of
        chest_region, chest_type, pairs is specified Region will superceed type,
        and type will superceed pairs.

    in_df: Pandas dataframe
        Contains feature information previously computed over the patches
        for which we seak the distance information    

    x_extent: int
        region size in the x direction over which the feature will
        be estimated. The region will be centered at the patch center.
            
    y_extent: int
        region size in the y direction over which the feature will
        be estimated. The region will be centered at the patch center.
        
    z_extent: int
        region size in the z direction over which the feature will
        be estimated. The region will be centered at the patch center.
 
                
    Attribues
    ---------
    df_ : Pandas dataframe
        Contains the patch feature information. The 'patch_label' column
        corresponds to the segmented patch over which the distance is 
        computed. The 'chestwall_distance' column contains the physical distance 
        from the center of the patch to the chest wall.        
    """
    def __init__(self, x_extent = 30, y_extent=30, z_extent=1, \
        chest_region=None, chest_type=None, pair=None, in_df=None):
        # get the region / type over which distance is computed

        self.df_ = in_df
        self.x_extent=int(x_extent)
        self.y_extent=int(y_extent)
        self.z_extent=int(z_extent)
        self.chest_region=chest_region
        self.chest_type=chest_type
        self.pair=pair
        
    def fit(self, distance_image, distance_header, lm, xml_object):
        """Compute the histogram of each patch defined in 'patch_labels' beneath
        non-zero voxels in 'lm' using the CT data in 'ct'.
        
        Parameters
        ----------
        xmlobject: xml object containing geometrytopology data
        
        distance_image: 3D numpy array, shape (L, M, N)
            Input distance image 
    
        ct_header: header information for the distance image. Should be a 
            dictionary with the following entries: origin, direction, spacing.
               
        lm: 3D numpy array, shape (L, M, N)
            Input mask where distance features wil be extracted.    
        """        
        #build transformation matrix
        the_origin = np.array(distance_header['space origin'])
        the_direction = np.reshape(np.array(distance_header['space directions']), [3,3])
        the_spacing = np.array(distance_header['spacing'])

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
        my_geometry_data = GeometryTopologyData.from_xml(xml_object)
        
        # loop through each point and create a patch around it
        inc = 1
        the_patch = np.zeros_like(lm)
        
        for the_point in my_geometry_data.points : 
            coordinates = the_point.coordinate
            # feature_type = the_point.feature_type
    
            #print ("point "+str(chest_region)+" "+str(chest_type)+" "+str(the_point.coordinate)+" "+str(feature_type))
            
            ijk_val = transformationMatrix.MultiplyPoint([coordinates[0],\
                coordinates[1],coordinates[2],1]) # need to append a 1 at th eend of point

            # from here we can build the patches ...       
            the_patch[int(ijk_val[0])-2:int(ijk_val[0])+3, int(ijk_val[1])- \
                2:int(ijk_val[1])+3,int(ijk_val[2])] = inc
            inc = inc+1
        self.dist_extractor = DistanceFeatureExtractor(chest_region=self.chest_region,
                                    in_df=self.df_,x_extent = self.x_extent, \
                                    y_extent=self.y_extent, z_extent=self.z_extent) 
        self.dist_extractor.fit( distance_image, lm, the_patch) # df will have region / type entries
        self.df_ = pd.DataFrame(columns=self.dist_extractor.df_.columns)
        self.df_ = self.dist_extractor.df_

                    
if __name__ == "__main__":
    desc = """Generates histogram features given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_dist',
                      help='Input distance map file', dest='in_dist', \
                      metavar='<string>', default=None)
    parser.add_option('--in_lm',
                      help='Input mask file. The histogram will only be \
                      computed in areas where the mask is > 0. If lm is not \
                      included, the histogram will be computed everywhere.', 
                      dest='in_lm', metavar='<string>', default=None)
    parser.add_option('-r',
                      help='Chest region indicating what structure the \
                      distance map corresponds to. Note that either a \
                      region, or a type, or a region-type pair must be \
                      specified.',
                      dest='chest_region', metavar='<string>', default=None)
    parser.add_option('-t',
                      help='Chest type indicating what structure the distance \
                      map corresponds to. Note that either a region, or a \
                      type, or a region-type pair must be specified.',
                      dest='chest_type', metavar='<string>', default=None)
    parser.add_option('-p',
                      help='Chest region-type pair (comma-separated tuple) \
                      indicating what structure the distance map corresponds \
                      to. Note that either a region, or a type, or a \
                      region-type pair must be specified.',
                      dest='pair', metavar='<string>', default=None)
    parser.add_option('--out_csv',
                      help='Output csv file with the features', dest='out_csv', 
                      metavar='<string>', default=None)
    parser.add_option('--in_xml',
                      help='xml input', dest='in_xml',
                      metavar='<string>', default=None)
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
    parser.add_option('--in_csv',
                      help='Input csv file with the existing features. The \
                      distance features will be appended to this.', 
                      dest='in_csv', metavar='<string>', default=None)  
    (options, args) = parser.parse_args()
    
    image_io = ImageReaderWriter()
    if options.in_dist is None:
        raise ValueError("Must specify as distance map")

    
    print ("Reading distance map...")
    distance_map, dm_header = image_io.read_in_numpy(options.in_dist) 

    with open(options.in_xml, 'r+b') as f:
        xml_data = f.read()
    
    if (options.in_lm is not None):
        print ("Reading mask...")
        lm,lm_header = nrrd.read(options.in_lm) 
    else:
         lm = np.ones(np.shape(distance_map))   
    
    if (options.in_csv is not None):
        print ("Reading previously computed features...")
        init_df = pd.read_csv(options.in_csv)
    else:    
        init_df = None

    if options.chest_region is None and options.chest_type is None and \
      options.pair is None:
      raise ValueError("Must specify a chest region, or chest type, or \
                        region-type pair that the distance map corresponds to")

    pair = None
    if options.pair is not None:
        tmp = options.pair.split(',')
        assert len(tmp) == 2, 'Specified pairs not understood'
        pair = [options.pair.split(',')[0],options.pair.split(',')[1] ]

    print ("Computing distance features...")
    dist_extractor = DistanceFeatureExtractorFromXML(chest_region=options.chest_region, \
        chest_type=options.chest_type, pair=pair, in_df=init_df, x_extent=options.x_extent, \
                                                     y_extent=options.y_extent, z_extent=options.z_extent)
    dist_extractor.fit(distance_map, dm_header, lm, xml_data)

    if options.out_csv is not None:
        print ("Writing...")
        dist_extractor.df_.to_csv(options.out_csv, index=False)

    print ("DONE.")
