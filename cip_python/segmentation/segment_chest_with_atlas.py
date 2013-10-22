import pdb

def segment_chest_with_atlas(input_image, priors, likelihood):
    """Segment structures using atlas data.    

    Parameters
    ----------
    input_image : array, shape (L, M, N)
        Input image in the form a 3D matrix

    priors : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.

    likelihood : Instance of type StructureLikelihood
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    # TODO: Describe step 1

    # TODO: Describe step 2

    # TODO: Describe step 3	

    pass
