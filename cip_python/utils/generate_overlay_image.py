import numpy as np
from scipy import ndimage



def generate_overlay_image(ct_slice, labelmap_slice, window_width= None, 
    window_level= None, opacity= None):
    """ This function that takes as input a 2D ct slice and a labelmap and 
    returns an overlay
 
    Parameters
    ----------
    ct_image : array, shape ( N, N )
        A description
    
    labelmap_slice : array, shape ( N, N )
        A description
    
    window_width : int
    
    window_level : int
    
    opacity : float. Not yet implemented
    
    
    Returns
    -------
    overlay : array, shape ( N, N, 3)
        Array with the r,g,b, pixel values of the overlayed image
    
    """
    assert ct_slice.shape[0] == labelmap_slice.shape[0], \
        "CT slice and label disagree in dimension"
    assert ct_slice.shape[1] == labelmap_slice.shape[1], \
        "CT slice and label disagree in dimension"
    
    length = labelmap_slice.shape[0]
    width = labelmap_slice.shape[1]

    overlayed_image = np.squeeze(ndimage.rotate(labelmap_slice, -90))
    ct_image = np.squeeze(ndimage.rotate(ct_slice, -90))
    
    #shift all values to positive
    min_ct = np.float(np.min(ct_image))
    ct_image = ct_image - min_ct
    
    if (window_width == None):
        max_ct = np.float(np.max(ct_image))
        window_width = max_ct 
        window_level = max_ct/2
    else: 
        window_level = window_level - min_ct
    
    rgbArray = np.zeros((length,width,3), 'uint8')
    grey_val = ((ct_image.astype(np.float))*256.0/\
        (window_width+window_level)).astype(np.uint8)
    rgbArray[..., 0] =  rgbArray[..., 1] = rgbArray[..., 2] = grey_val
    color_val = overlayed_image>0
    rgbArray[color_val, 0] = 255
    
    return rgbArray
    