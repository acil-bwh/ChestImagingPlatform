import numpy as np
from skimage.segmentation import slic
from optparse import OptionParser
import nrrd

import pdb
#from cip_python.io.image_reader_writer import ImageReaderWriter
     
class SlicSegmenter:
    """General purpose class interfacing to the SLIC segmenter. 

    The user inputs a CT image to be segmented. The output is is a 3D numpy 
    array where for each voxel, a label is assigned designating the patch index
    that the voxel belongs to. 
    

    
    Parameters (SLIC parameters)
    ----------
    n_segments : int, optional
        The (approximate) number of labels in the segmented output image.

    compactness : float, optional
        Balances color-space proximity and image-space proximity. Higher \
        values give more weight to image-space. As compactness tends to \
        infinity, superpixel shapes become square/cubic. In SLICO mode, this \
        is the initial compactness.

    max_iter : int, optional
        Maximum number of iterations of k-means.
    
    sigma : float or (3,) array-like of floats, optional
        Width of Gaussian smoothing kernel for pre-processing for each \
        dimension of the image. The same sigma is applied to each dimension in \
        case of a scalar value. Zero means no smoothing. Note, that sigma is \
        automatically scaled if it is scalar and a manual voxel spacing is \
        provided (see Notes section).

    spacing : (3,) array-like of floats, optional
        The voxel spacing along each image dimension. By default, slic assumes \
        uniform spacing (same voxel resolution along z, y and x). This \
        parameter controls the weights of the distances along z, y, and x \
        during k-means clustering.

    enforce_connectivity: bool, optional (default False)
        Whether the generated segments are connected or not

    min_size_factor: float, optional
        Proportion of the minimum segment size to be removed with respect to \
        the supposed segment size `depth*width*height/n_segments`

    max_size_factor: float, optional
        Proportion of the maximum connected segment size. A value of 3 works \
        in most of the cases.

    slic_zero: bool, optional
        Run SLIC-zero, the zero-parameter mode of SLIC. [R337]
    
    ct: 3D numpy array, shape (L, M, N)
        Input CT image to be segmentaed
                
    Returns
    --------
    segmentation: 3D numpy array, shape (L, M, N)
        Contains a label assignment for each CT voxel
        
    """        
    def __init__(self, n_segments=100, compactness=10., max_iter=10, sigma=0,
                 spacing=None, enforce_connectivity=False, min_size_factor=0.5, 
                 max_size_factor=3, slic_zero=False):        
        self.n_segments =  n_segments    
        self.compactness = compactness
        self.max_iter = max_iter
        self.sigma = sigma
        self.enforce_connectivity = enforce_connectivity
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor
        self.slic_zero = slic_zero
        self.spacing=spacing
                
    def execute(self, ct):
        segmentation = slic(ct, multichannel=False, n_segments=self.n_segments, \
            compactness=self.compactness, max_iter=self.max_iter, \
            sigma=self.sigma, spacing=self.spacing, \
            enforce_connectivity=self.enforce_connectivity, \
            min_size_factor=self.min_size_factor, \
            max_size_factor=self.max_size_factor, slic_zero=self.slic_zero)

        return segmentation        

if __name__ == "__main__":
    desc = """Generates parenchyma phenotypes given input CT and segmentation \
    data"""
    
    parser = OptionParser(description=desc)
    parser.add_option('--in_ct',
                      help='Input CT file', dest='in_ct', metavar='<string>',
                      default=None)
    parser.add_option('--out_lm',
                      help='Output SLIC segmentation', dest='out_lm', 
                      metavar='<string>', default=None)
    parser.add_option('--n_segments',
                      help='SLIC param. The (approximate) number of labels in \
                      the segmented output image. (optional)', dest='n_segments', 
                      metavar='<string>', default=100)
    parser.add_option('--compactness',
                      help='SLIC param. Balances color-space proximity and \
                      image-space proximity. Higher values give more weight to \
                      image-space. As compactness tends to infinity, \
                      superpixel shapes become square/cubic.  (optional)', 
                      dest='compactness', metavar='<string>', default=10.)
    parser.add_option('--max_iter',
                      help='SLIC param. Maximum number of iterations of \
                      k-means.  (optional)',  dest='max_iter', 
                      metavar='<string>', default=10)
    parser.add_option('--sigma',
                      help='SLIC param. Width of Gaussian smoothing kernel for \
                      pre-processing for each dimension of the image. The same \
                      sigma is applied to each dimension in case of a scalar \
                      value.  (optional)',  dest='sigma', \
                      metavar='<string>', default=0.)
    parser.add_option('--spacing',
                      help='SLIC param.  The voxel spacing along each image \
                      dimension. By default, slic assumes uniform spacing \
                      (same voxel resolution along z, y and x). This parameter \
                      controls the weights of the distances along z, y, and x \
                      during k-means clustering. (optional).',  dest='spacing', 
                      metavar='<string>', default=None)                      
    parser.add_option('--enforce_connectivity',
                      help='Whether the generated segments are connected or \
                      not',  dest='enforce_connectivity', action="store_true", \
                      default=False)                              
    parser.add_option('--min_size_factor',
                      help='SLIC param.  Proportion of the minimum segment \
                      size to be removed with respect to the supposed segment \
                      size `depth*width*height/n_segments`(optional).', 
                      dest='min_size_factor', metavar='<string>', default=0.5)    
    parser.add_option('--max_size_factor',
                      help='SLIC param.  Proportion of the maximum connected \
                      segment size. A value of 3 works in most of the cases.\
                      (optional).', dest='max_size_factor', \
                      metavar='<string>', default=3.)  
    parser.add_option('--slic_zero',
                      help='Run SLIC-zero, the zero-parameter mode of SLIC. \
                      http://ivrgwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/',  
                      dest='slic_zero', action="store_true", default=False)  


    (options, args) = parser.parse_args()
    
    #image_io = ImageReaderWriter()
    ct, ct_header = nrrd.read(options.in_ct) #image_io.read_in_numpy(options.in_ct)

    # Preprocess the CT to enhance contrast within the lung
    ct[ct > -600] = 0    
    m = 256/(700.)
    b = 1000.*m
    ct = (m*ct + b).clip(0, 255)
    
    spacing = np.zeros(3)
    spacing[0] = ct_header['space directions'][0][0]
    spacing[1] = ct_header['space directions'][1][1]
    spacing[2] = ct_header['space directions'][2][2]

    slic_segmenter = SlicSegmenter(n_segments=int(options.n_segments), \
        compactness=float(options.compactness), max_iter=options.max_iter, \
        sigma=options.sigma, spacing=options.spacing, \
        enforce_connectivity=options.enforce_connectivity, \
        min_size_factor=options.min_size_factor, \
        max_size_factor=options.max_size_factor, slic_zero=options.slic_zero)

    slic_segmentation = slic_segmenter.execute(ct)

    #slic_header = {}
    #slic_header['dimension'] = 3
    #slic_header['kinds'] = ['domain', 'domain', 'domain']
    #slic_header['sizes'] = [512, 512, 502]
    #slic_header['space'] = 'left-posterior-superior'
    #slic_header['space directions'] = np.array([[0.664062, 0.0, 0.0],
    #                                            [0.0, 0.664062, 0.0],
    #                                            [0.0, 0.0, 0.625]])
    #slic_header['space origin'] = np.array([-169.7, -164.2, -319.875])

    
    nrrd.write(options.out_lm, slic_segmentation)
    #image_io.write_from_numpy(slic_ct_segmentation,ct_header,options.out_lm)
        

    #    imshow(rotate(mark_boundaries(ct[:,:,256], slic_segmentation[:,:,256]), -90) , cmap = cm.Greys_r  )
    #imshow(rotate((ct[:,:,256]*m + b).clip(0, 255), -90), cmap = cm.Greys_r)
