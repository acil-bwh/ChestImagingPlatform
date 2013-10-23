import pdb

from chest_structure_likelihoods import ChestStructureLikelihood

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

    likelihoods : List StructureLikelihood class instances
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    # Step 1: For all structures of interest, compute the posterior energy
    posterior_energies = compute_structure_posterior_energy(likelihood, priors)
    
    # Step 3: For each structure separately, input the posterior energies into the graph cuts code
    
    #Step 4: Postprocess by removing duplicate labels and pre-segmented structures (e.g. trachea)
    
    pass

def compute_structure_posterior_energy(likelihood, priors):
    """Computes the posterior energy given a list of structure likelihoods
       and priors.  
        Parameters
    ----------
    priors : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.

    likelihoods : List StructureLikelihood class instances
        ...
        
    Returns
    -------
    energies : List of float arrays with shape (L, M, N) representing posterior 
              energies for each structure/non structure
    """
    
    pass
    
def obtain_graph_cuts_segmentation(structure_posterior_energy, not_structure_posterior_energy):
    """Obtains the graph cuts segmentation for a structure given the posterior energies.  
    
    Parameters
    ----------
    structure_posterior_energy: A float array with shape (L, M, N) 
            representing the posterior energies for the structure of interest. 
    not_structure_posterior_energy :  A float array with shape (L, M, N) 
            representing the posterior energies for not being the structure
            of interest. 
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    pass