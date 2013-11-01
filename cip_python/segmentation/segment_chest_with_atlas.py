import pdb
import sys
sys.path.append("/Users/rolaharmouche/ChestImagingPlatformPrivate/cip_python/utils")
import weighted_feature_map_densities
import numpy as np
import construct_chest_atlas

def segment_lung_with_atlas(input_image, probabilistic_atlas):
    """Segment lung using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlas : float array, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of

    list_of_label_files : list of file names (string)
        Each file name points to a file that has a labeled 
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    # Threshold atlas

    # Compute distance map using thresholded atlas

    # Compute likelihood
    
    #segment given priors
    segment_chest_with_atlas(input_image, prior, likelihood)

def segment_chest_with_atlas(input_image, priors):
    """Segment structures using atlas data. Computes the likelihoods given a
       feature_map, an input feature vector and priors. Uses exponential likelihood    

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    prior : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    #Compute feature vector
    feature_vector = compute_intensity_distance_feature_vector(input_image, priors)
    
    #Define likelihood and compute it, for each label
    num_classes = priors.len
    likelihoods = np.zeros(np.size(priors))
    
    my_polynomial_feature_map = weighted_feature_map_densities.PolynomialFeatureMap( feature_vector,[2] )  
    my_polynomial_feature_map.compute_num_terms()

    for d in range(0, num_classes):
        posteriors[d] = likelihoods[d]*priors[d]/np.sum(likelihoods*priors)
        
    #Call segment_chest_with_atlas with likelihood an dprior
    
    pass
    
def compute_intensity_distance_feature_vector(input_image, priors):
    """Obtain the intensity and distance from lung feature vector.    

    Parameters
    ----------
    input_image : array, shape (L, M, N)
        Input image in the form a 3D matrix

    priors : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.

        ...
        
    Returns
    -------
    feature_vector : list of float arrays with shape (L, M, N)
        Each feature will be represented by an array having the
	same size as the input image.
    """    
    
    intensity_distance_feature = input_image
    return intensity_distance_feature

def segment_chest_with_atlas(input_image, priors, likelihoods):
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

    likelihoods : List WeightedFeatureMapDensity class instances
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    # Step 1: For all structures of interest, compute the posterior energy 
           
    posterior_energies = compute_structure_posterior_energies(likelihoods, priors)
    
    # Step 3: For each structure separately, input the posterior energies into the graph cuts code
    for i in range(priors.len):
         label_map=obtain_graph_cuts_segmentation(posterior_energies[i*2],posterior_energies[i*2+1])
    #Step 4: Postprocess by removing duplicate labels and pre-segmented structures (e.g. trachea)
    
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
