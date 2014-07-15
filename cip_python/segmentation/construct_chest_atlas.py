import numpy as np



def construct_probabilistic_atlas(label_maps, normalize=True, weights=None,
                                  atlas=None):
    """Creates probabilistic atlas from a collection of label maps

    Parameters
    ----------
    label_maps : List of arrays, shape (L, M, N)
        Each element of the list is an LxMxN array with the structure of
        interest isolated. Any non-zero value will be considered as part
        of the structure of interest. It is assumed that all label maps
        have been registered to the same coordinate frame.

    normalize : boolean, optional
        If true, the returned atlas will be the sum of the inputs divided by
        the number of input label maps

    weights : float array, shape Dx1, optional 
        Each of the label maps in specified in 'label_maps' can be weighted
        by the corresponding weights provided in this vector. If not
        specified, all label maps will we weighted equally.

    in_atlas : array, shape (L, M, N)
        The returned atlas will be computed as described above and then added
        to this array before returning. Useful for repeated calls to this
        function.
        
    Returns
    -------
    out_atlas : array, shape (L, M, N)
        Output atlas
    """
    # TODO: check that All images in 'label_maps' must be the same size

    # TODO: check that If 'weights' not None, 
    # test that each weight is between 0 and 1
    
    #initialise atlas if NONE

    if atlas is None:
         #atlas = np.zeros([np.shape(label_maps[0])[0], np.shape( \
         #       label_maps[0])[1], np.shape(label_maps[0])[2]], dtype=float)
         atlas = np.zeros(np.shape(label_maps[0]), dtype=float)        
    
    # Get the number of labelmaps
    num_labelmaps = len(label_maps)
    #for number of labelmaps, add label_maps*weight to atlas
    for i in range(0, num_labelmaps):
        atlas = np.add(atlas, np.multiply(label_maps[i].astype(np.float), \
                weights[i].astype(np.float)))
        
    return atlas


    


    
