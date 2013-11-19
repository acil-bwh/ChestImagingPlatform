import pdb
import sys
import numpy as np
from cip_python.utils.weighted_feature_map_densities \
    import ExpWeightedFeatureMapDensity
from pygco import cut_from_graph

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
    label_map : list of integer array, shape (L, M, N)
        Each segmented strcture of interest will be represented by an array 
        with binary labels.
    """
    # Step 1: For all structures of interest, compute the posterior energy 
    
    #Define likelihood and compute it, for each label
    num_classes = priors.len
    likelihoods = np.zeros(np.size(priors), dtype=np.float)
    posterior_probabilities = np.zeros(np.size(priors), dtype=np.float)
    label_map=np.zeros(np.size(priors), dtype=np.int)
    
    my_polynomial_feature_map = \
      ExpWeightedFeatureMapDensity.PolynomialFeatureMap(feature_vector, [2])
    my_polynomial_feature_map.compute_num_terms()

    #listoflists=[ [0]*4 ] *5
    posterior_probabilities = \
       compute_structure_posterior_probabilities(likelihoods, priors)
        
    # Step 3: For each structure separately, input the posterior energies into
    # the graph cuts code    
    for i in range(num_classes):
        class_posterior_energies= posterior_probabilities[i]
        not_class_posterior_energies = 1 - posterior_probabilities[i]
        label_map[i]=obtain_graph_cuts_segmentation(class_posterior_energies, \
          not_class_posterior_energies)
            
    return label_map



def compute_structure_posterior_probabilities(likelihoods, priors):
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
            representing the posterior energies for the structure of interest. (source) 
    not_structure_posterior_energy :  A float array with shape (L, M, N) 
            representing the posterior energies for not being the structure
            of interest. (sink)
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    length = np.shape(structure_posterior_energy)[0];
    width = np.shape(structure_posterior_energy)[1];
    num_slices = np.shape(structure_posterior_energy)[2];
    numNodes = length * width
    segmented_image = np.zeros((length, width, num_slices), dtype = np.int32)
    
    for slice_num in range(0, num_slices):
    
        source_slice = structure_posterior_energy.ravel()[length*width*slice_num: length*width*(slice_num+1)].squeeze().reshape(length,width).astype(np.int32) 
        sink_slice = not_structure_posterior_energy.ravel()[length*width*slice_num: length*width*(slice_num+1)].squeeze().reshape(length,width).astype(np.int32) 
  
        imageIndexArray =  np.arange(numNodes).reshape(np.shape(source_slice)[0], np.shape(source_slice)[1])
 
        #Adding neighbourhood terms 
        inds = np.arange(imageIndexArray.size).reshape(imageIndexArray.shape) #goes from [[0,1,...numcols-1],[numcols, ...],..[.., num_elem-1]]
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()] #all rows, not last col make to 1d
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()] #all rows, not first col, make to 1d
        edges = np.vstack([horz, vert]).astype(np.int32) #horz is first element, vert is 
        theweights = np.ones((np.shape(edges))).astype(np.int32)*18
        edges = np.hstack((edges,theweights))[:,0:3].astype(np.int32) #stack the weight value hor next to edge indeces
    
        #3rd order neighbours
        horz = np.c_[inds[:, :-2].ravel(), inds[:,2:].ravel()] #all rows, not last col make to 1d
        vert = np.c_[inds[:-2, :].ravel(), inds[2:, :].ravel()] #all rows, not first col, make to 1d
        edges2 = np.vstack([horz, vert]).astype(np.int32) #horz is first element, vert is 
        theweights2 = np.ones((np.shape(edges2))).astype(np.int32)
        edges2 = np.hstack((edges2,theweights2))[:,0:3].astype(np.int32)
    
        edges = np.vstack([edges,edges2]).astype(np.int32)

        pairwise_cost = np.array([[0, 1], [1, 0]], dtype = np.int32)
    
        energies = np.dstack((np.array(source_slice).astype(np.int32).flatten(), np.array(sink_slice).astype(np.int32).flatten())).squeeze()

        segmented_slice = cut_from_graph(edges, energies, pairwise_cost, 3,'expansion') 
        segmented_image[:,:,slice_num] = segmented_slice.reshape(length,width).transpose()

    return segmented_image
