import pdb
import sys
import numpy as np
from cip_python.utils.weighted_feature_map_densities \
    import ExpWeightedFeatureMapDensity
from cip_python.segmentation.construct_chest_atlas \
    import construct_probabilistic_atlas
import construct_chest_atlas

def segment_chest_with_atlas(input_image, feature_vector, priors):
    """Segment structures using atlas data. Computes the likelihoods given a
    feature_map, an input feature vector and priors. Uses exponential
    likelihood

    Parameters
    ----------
    input_image : float array, shape (L, M, N)
        TODO

    priors : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.
        
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    # Step 1: For all structures of interest, compute the posterior energy 
    
    #Define likelihood and compute it, for each label
    num_classes = priors.len
    likelihoods = np.zeros(np.size(priors))
    
    my_polynomial_feature_map = \
      weighted_feature_map_densities.PolynomialFeatureMap(feature_vector, [2])
    my_polynomial_feature_map.compute_num_terms()

    #listoflists=[ [0]*4 ] *5
    posterior_energies = \
       compute_structure_posterior_energies(likelihoods, priors)
        
    # Step 3: For each structure separately, input the posterior energies into
    # the graph cuts code    
    for i in range(num_classes):
        label_map=obtain_graph_cuts_segmentation(posterior_energies[i*2],posterior_energies[i*2+1])
            
    return label_map



def compute_structure_posterior_energies(likelihoods, priors):
    """Computes the posterior energy given a list of structure likelihoods
       and priors.  

    Parameters
    ----------
    priors : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.

    likelihoods : List WeightedFeatureMapDensity class instances
        ...
        
    Returns
    -------
    energies : List of float arrays with shape (L, M, N) representing posterior 
              energies for each structure/non structure
    """
    
    #get the number of classes, initialize list of posteriors
    num_classes = likelihoods.len
    assert num_classes == priors.len
    posteriors = np.zeros(np.size(likelihoods))
    
    #for each class posteriors[i] = likelihoods[i]*priors[i]/sum(likelihoods[i]*priors[i])
    for d in range(0, num_classes):
        posteriors[d] = likelihoods[d]*priors[d]/np.sum(likelihoods*priors)
 
    return posteriors
    
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
