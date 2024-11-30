from optparse import OptionParser

import numpy as np
import pandas as pd
from cip_python.input_output import ImageReaderWriter

class LabelsReader(object):

    def apply_reader_labels(self, seg, seg_header, features_df, plocs_df=None,
                            ilocs_df=None):
        """Apply reader labels stored in the region-type locations dataframe to
        the features dataframe. Note that features_df is modified in place.

        Parameters
        ----------
        seg : numpy array, shape ( I, J, K )
            Array containing the segmentation information. This array is used to
            generate the mapping between the coordinates in the region-type
            locations dataframe to the patch labels in the features dataframe. Note
            that the same patches (segmentation) volume must be used here as was
            used to generate the data in the features file.

        seg_header : nrrd header
            Header information corresponding to the segmentation volume. This header
            information is used to map between physical coordinates contained in the
            plocs_df (if specified), to an index location in the segmentation
            volume.

        features_df : pandas dataframe
            Contains the feature vectors that the labels will be assigned to.

        plocs_df : pandas dataframe, optional
            Contains reader-assigned labels in the ChestRegion and ChestType columns
            and corresponding physical (world) coordinates. Note that either a
            plocs_df or an ilocs_df must be specified.

        ilocs_df : pandas dataframe, options
            Contains reader-assigned labels in the ChestRegion and ChestType columns
            and corresponding image index coordinates. Note that either a plocs_df
            or an ilocs_df must be specified.
        """
        if plocs_df is None and ilocs_df is None:
          raise ValueError("Must specify either plocs_df or ilocs_df")

        if plocs_df is not None and ilocs_df is not None:
          raise ValueError("Must specify either plocs_df or ilocs_df, not both")

        # If a plocs_df is specified, create an ilocs_df from it
        if plocs_df is not None:
            sp_x = seg_header['spacing'][0]
            sp_y = seg_header['spacing'][1]
            sp_z = seg_header['spacing'][2]

            or_x = None
            or_y = None
            or_z = None
            for k in seg_header.keys():
                if 'origin' in k:
                    or_x = seg_header[k][0]
                    or_y = seg_header[k][1]
                    or_z = seg_header[k][2]
            assert or_x is not None, "Origin not retrieved from header"

            ilocs_df = \
              pd.DataFrame(columns=['Region', ' Type', \
                                    ' X index', ' Y index', ' Z index'])
            for i in xrange(0, plocs_df.shape[0]):
                cip_region = plocs_df.ix[i, 'Region']
                cip_type = plocs_df.ix[i, ' Type']
                x_index = np.int((plocs_df.ix[i, ' X point'] - or_x)/sp_x)
                y_index = np.int((plocs_df.ix[i, ' Y point'] - or_y)/sp_y)
                z_index = np.int((plocs_df.ix[i, ' Z point'] - or_z)/sp_z)
                ilocs_df.loc[i] = [cip_region, cip_type, x_index, y_index, z_index]

        for n in xrange(0, ilocs_df.shape[0]):
            i = ilocs_df.ix[n, ' X index']
            j = ilocs_df.ix[n, ' Y index']
            k = ilocs_df.ix[n, ' Z index']
            seg_value = seg[i, j, k]

            cip_region = ilocs_df.ix[n, 'Region']
            cip_type = ilocs_df.ix[n, ' Type']

            index = features_df['patch_label'] == seg_value
            features_df.ix[index, 'ChestRegion'] = cip_region
            features_df.ix[index, 'ChestType'] = cip_type
        
if __name__ == "__main__":
    desc = """Apply reader labels stored in the region-type locations file to \
    the input features file"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_seg',
                      help='Input segmentation file. This file is used to \
                      generate the mapping between the coordinates in the \
                      region-type locations file to the patch labels in the \
                      features file. Note that the same pathes (segmentation) \
                      volume must be used here as was used to generate the \
                      data in the features file', dest='in_seg', 
                      metavar='<string>', default=None)
    parser.add_option('--features',
                      help='Input features file. Once read, these features \
                      will be updated with the reader-assigned chest-region \
                      and chest-type labels contained in the region-type \
                      locations file.', dest='features', 
                      metavar='<string>', default=None)    
    parser.add_option('--ilocs',
                      help='Region-type locations file containing \
                      reader-assigned labels and corresponding image index \
                      coordinates. Either an index file or a points file must \
                      be specified.', dest='ilocs', metavar='<string>', \
                      default=None)    
    parser.add_option('--plocs',
                      help='Region-type locations file containing \
                      reader-assigned labels and corresponding physical point \
                      coordinates. Either an index file or a points file must \
                      be specified.', dest='ilocs', metavar='<string>', \
                      default=None)               
    parser.add_option('--labeled',
                      help='File name for output, labeled features file.', \
                      dest='labeled', metavar='<string>', default=None)                                                    

    (options, args) = parser.parse_args()

    if options.in_seg is None:
        raise ValueError("Must specify a segmentation file")
    if options.features is None:
        raise ValueError("Must specify a features file")
    if options.ilocs is None and options.plocs is None:
        raise ValueError("Must specify a chest-region chest-type file")    
    
    print ("Reading segmentation...")
    image_io = ImageReaderWriter()
    seg, seg_header = image_io.read_in_numpy(options.in_seg) 

    print ("Reading features file...")
    features_df = pd.read_csv(options.features)

    print ("Reading labels...")
    plocs_df = None
    ilocs_df = None
    if options.ilocs is not None:
        ilocs_df = pd.read_csv(options.ilocs)
    else:
        plocs_df = pd.read_csv(options.ilocs)

    print ("Applying reader labels...")
    reader = LabelsReader()
    reader.apply_reader_labels(seg, seg_header, features_df, plocs_df, ilocs_df)

    if options.labeled is not None:
        print ("Writing labeled features...")
        features_df.to_csv(options.labeled)
