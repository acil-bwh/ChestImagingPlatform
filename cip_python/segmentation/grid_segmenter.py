import numpy as np
from optparse import OptionParser
from ..input_output import ImageReaderWriter
     
class GridSegmenter:
    """Segments a volume into a grid of volume patches. 

    The user inputs a CT image to be segmented or the shape of the volume for
    which to provide a grid segmentation. The output is is a 3D numpy 
    array where for each voxel, a label is assigned designating the patch index
    that the voxel belongs to. The way that it is presently done is that there
    is a z_offset, and a 30 by 30 patch is extracted at each z location 
    0:z_offset:max.
    

    
    Parameters 
    ----------

    input_size : int array, shape 3x1, optional 
        x,y,z size of the volume for which to form a patch grid
        either this or ct need to be set

    ct: 3D numpy array, shape (L, M, N)
        Input CT image to be segmentaed
                
    xy_patch_size : int, optional
        The x-y dimension of each patch. Default: 30
        
    z_patch_offset : the z offset between consecutive patches. Default : 10      
               
    Returns
    --------
    segmentation: 3D numpy array, shape (L, M, N)
        Contains a label assignment for each CT voxel
        
    """
        
    def __init__(self, input_dimensions=None, ct=None, x_size=31, y_size=31,
        z_offset=10):        
        self.input_dimensions =  input_dimensions    
        self.ct = ct
        self.x_size = x_size
        self.y_size = y_size
        self.z_offset = z_offset

    def execute(self):        
        assert (self.input_dimensions is not None) or  (self.ct is not None), \
            "Either CT or input shape need to be set"  

        assert (self.input_dimensions is None) or  (self.ct is None), \
            "CT and input shape cannot both be set" 
        
        if (self.ct is not None):
            # get array size
            ct_shape = np.shape(self.ct) 
        else:
            ct_shape = self.input_dimensions
                                                                                                                                                                                                                                   
        assert(ct_shape[0]>= self.x_size) and (ct_shape[1]>= self.y_size) and \
            (ct_shape[2]>= self.z_offset), "image dimensions "+str(ct_shape)+"must be larger than  \
            grid size: "+str(self.x_size)+" "+str(self.y_size)+" "+str(self.z_offset)
        
        segmentation = np.zeros((ct_shape[0], ct_shape[1],ct_shape[2]))
                                #, dtype=np.int)
        [gridX,gridY, gridZ] = np.meshgrid(np.unique(np.append(
            np.arange(0, ct_shape[0], self.x_size), ct_shape[0])), \
            np.unique(np.append(np.arange(0, ct_shape[1], self.y_size), \
            ct_shape[1])), np.unique(np.append(np.arange(0, ct_shape[2], \
            self.z_offset), ct_shape[2])));
        # go through all elements in gridX and segment 
        #([gridX-15,gridX+15],[gridY-15,gridY+15])
        patch_id = 1    
        
        for k in np.arange(0, np.shape(gridX)[2]-1):
            for j in np.arange(0, np.shape(gridX)[1]-1):
                for i in np.arange(0, np.shape(gridX)[0]-1): #x,y,z
                    segmentation[int(np.floor(gridX[i,j,k])):int(np.floor( \
                        gridX[i+1,j+1,k+1])),int(np.floor(gridY[i,j,k])):int(np.floor(\
                        gridY[i+1,j+1,k+1])),int(np.floor(gridZ[i,j,k])):int(np.floor(\
                        gridZ[i+1,j+1,k+1]))]= patch_id
                    patch_id = patch_id+1
    
        return segmentation
        
if __name__ == "__main__":
    desc = """Generates parenchyma phenotypes given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file.', dest='in_ct', 
                      metavar='<string>', default=None)
    parser.add_option('--out_seg',
                      help='Output grid segmentation', dest='out_seg', 
                      metavar='<string>', default=None)
    parser.add_option('--xsize',
                      help='x dimensions of each patch (optional)', \
                      dest='x_size', metavar='<string>', default=30)
    parser.add_option('--ysize',
                      help='y dimensions of each patch (optional)', \
                      dest='y_size', metavar='<string>', default=30)                                     
    parser.add_option('--zoffset',
                      help='offset between consecutive slices of patches. This \
                      is essentially the z dimension of the patch (optional)', 
                      dest='z_offset', metavar='<string>', default=10)                  
    #parser.add_option('--dim',
    #                  help='Dimensions of the desired grid volume (optional). \
    #                  This can be input instead of the ct image. ',  
    #                  dest='dim', metavar='<string>', default=None)                      


    (options, args) = parser.parse_args()
    
    ct = None
    
    if (options.in_ct is not None):
        image_io = ImageReaderWriter()
        ct,ct_header=image_io.read_in_numpy(options.in_ct)
        #ct, ct_header=nrrd.read(options.in_ct)
       
    grid_segmentor = GridSegmenter(input_dimensions=None, ct=ct, \
        x_size=int(options.x_size), y_size=int(options.y_size), \
        z_offset=int(options.z_offset))
                    
    grid_ct_segmentation = grid_segmentor.execute()

    image_io.write_from_numpy(grid_ct_segmentation,ct_header,options.out_lm)
    #nrrd.write(options.out_seg, grid_ct_segmentation, ct_header)
        
