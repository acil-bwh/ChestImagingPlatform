import numpy as np
from scipy import ndimage
from pygco import cut_from_graph
from cip_python.utils.weighted_feature_map_densities \
    import ExpWeightedFeatureMapDensity
from cip_python.utils.feature_maps import PolynomialFeatureMap


#def segment_chest_with_atlas(feature_vector, feature_maps, priors):
def segment_chest_with_atlas(likelihoods, priors, normalization_constants):
    """Segment structures using atlas data and likelihoods. 

    Parameters
    ----------
    likelihoods : list of float arrays, shape (L, M, N)
        Likelihood values for each structure of interest
        
    priors : list of float arrays with shape (L, M, N)
        Each structure of interest will be represented by an array having the
	same size as the input image. Every voxel must have a value in the
	interval [0, 1], indicating the probability of that particular
	structure being present at that particular location.
        
    normalization_constants : list of float arrays with shape (L, M, N)
        Constant for each voxel and each class in order to render the output
        a true posterior probability.
        
    Returns
    -------
    label_map : list of integer array, shape (L, M, N)
        Each segmented strcture of interest will be represented by an array 
        with binary labels.
    """
    
    # For all structures of interest, compute the posterior energy 
    
    num_classes = np.shape(priors)[0]
    posterior_probabilities = np.zeros(np.shape(priors), dtype=np.float)
    label_map=np.zeros(np.shape(priors), dtype=np.int)
    
    posterior_probabilities = \
       compute_structure_posterior_probabilities(likelihoods, priors, \
                                                  normalization_constants)
                                                  
    import nrrd
    nrrd.write("/Users/rolaharmouche/Documents/Data/COPDGene/test_left_posterior.nrrd", posterior_probabilities[0])
    nrrd.write("/Users/rolaharmouche/Documents/Data/COPDGene/test_right_posterior.nrrd", posterior_probabilities[1])
    
    #  For each structure separately, input the posterior energies into
    # the graph cuts code and obtain a segmentation   
    for i in range(num_classes):
        not_class_posterior_energies= (posterior_probabilities[i].squeeze()* \
          4000).astype(np.int32)
        class_posterior_energies = ( (np.ones(np.shape( \
          not_class_posterior_energies)).astype(np.float)- \
          posterior_probabilities[i].squeeze())*4000).astype(np.int32)
        label_map[i]=obtain_graph_cuts_segmentation( \
          not_class_posterior_energies, class_posterior_energies)
    
    for i in range(0, num_classes):
        label_map[i] = ndimage.binary_fill_holes(label_map[i]).astype(int)
        label_map[i] = ndimage.binary_fill_holes(label_map[i]).astype(int)      
    #remove overlap between labels by choosing the label with the least energy
    
    total_labels = np.zeros([np.shape(label_map[0])[0], np.shape( \
       label_map[0])[1], np.shape(label_map[0])[2]], dtype=int)
    isMultiple_labels = np.zeros([np.shape(label_map[0])[0], \
       np.shape(label_map[0])[1], np.shape(label_map[0])[2]], dtype=int)
    lowest_energy_label = np.zeros([np.shape(label_map[0])[0], \
       np.shape(label_map[0])[1], np.shape(label_map[0])[2]], dtype=int)
    for i in range(0, num_classes):
        total_labels = np.add(total_labels, label_map[i])
    isMultiple_labels = (total_labels >1)
    
    #find lowest energy label, right now maximum posterior probability
    lowest_energy_label = np.argmax(posterior_probabilities, axis=0)

    #Debug: save lowest energy label
    nrrd.write("/Users/rolaharmouche/Documents/Data/COPDGene/14988Y/14988Y_INSP_STD_UAB_COPD/lowest_energy_label.nrrd", lowest_energy_label)

    
    #isMultiple_labels&lowest or nonmultiple&current
    #or multiple & (non-lowest) = 0

        
    for i in range(0, num_classes):    
        isNotLowestEnergyLabel = (i != lowest_energy_label)       
        label_map[i][(np.logical_and(isMultiple_labels,isNotLowestEnergyLabel))] = 0 
    

    return label_map


def segment_lung_with_atlas(input_image, probabilistic_atlases): 
    #TODO: and command line option to this command
    
    """Segment lung using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlases : list of float arrays, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of the class.
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    #compute feature vectors for left and right lungs
    ### replace with likelihood params for class 0.


    length  = np.shape(probabilistic_atlases[0])[0]
    width = np.shape(probabilistic_atlases[0])[1]
    
    lungPrior = probabilistic_atlases[0] + probabilistic_atlases[1]
    zero_indeces_thresholding = lungPrior < 0.35 
    lungPriorSlicedialated = lungPrior
    lungPriorSlicedialated[zero_indeces_thresholding] = 0.0
    
    ones_indeces_thresholding = lungPrior > 0.34 
    lungPriorSlicedialated[ones_indeces_thresholding] = 1.0

    lungPriorSlicedialated = ndimage.binary_dilation(lungPriorSlicedialated, \
      iterations=2)
    ndimage.morphology.binary_fill_holes(lungPriorSlicedialated, \
      structure=None, output=lungPriorSlicedialated, origin=0)
    tozero_indeces_intensities = lungPriorSlicedialated < 1
    
    #compute distance using opposite atlas binary map. Assumption
    zero_indeces_ll = probabilistic_atlases[0] < 0.5
    leftLungPriorthres = np.ones((length, width,np.shape( \
      probabilistic_atlases[0])[2])).astype(np.float)
    leftLungPriorthres[zero_indeces_ll] = 1.0    
    one_indeces_ll = probabilistic_atlases[0] >= 0.5
    leftLungPriorthres[one_indeces_ll] = 0.0      
    left_lung_distance_map = ndimage.morphology.distance_transform_edt( \
      leftLungPriorthres)
    
    zero_indeces_rl = probabilistic_atlases[1] < 0.5
    rightLungPriorthres = np.ones((length, width,np.shape( \
      probabilistic_atlases[1])[2])).astype(np.float)
    rightLungPriorthres[zero_indeces_rl] = 1.0    
    one_indeces_rl = probabilistic_atlases[1] >= 0.5
    rightLungPriorthres[one_indeces_rl] = 0.0      
    right_lung_distance_map = ndimage.morphology.distance_transform_edt( \
      rightLungPriorthres)
    
    left_polynomial_feature_map = PolynomialFeatureMap( [input_image, \
      left_lung_distance_map],[0,1,2] )  
    left_polynomial_feature_map.compute_num_terms()
    
    right_polynomial_feature_map = PolynomialFeatureMap( [input_image, \
      right_lung_distance_map],[0,1,2] )  
    right_polynomial_feature_map.compute_num_terms()

    #define the weights
    
    #exp(-(alpha(1)*Ival + alpha_(2)*Dval+alpha(3)))^2
    # = exp(-(alpha(1)^2*Ival^2 + alpha(1)*Ival*alpha(2)*Dval  \
    #    +    alpha(1)*Ival*alpha(3) +  alpha(2)*Dval * alpha(1)*Ival
    #    +alpha(2)*Dval*alpha(2)*Dval+alpha(2)*Dval *alpha(3)  \
    #    +  alpha(3)*alpha(1)*Ival+  alpha(3)*alpha(2)*Dval+alpha(3)*alpha(3)))
    
    # = exp(-(alpha(3)*alpha(3) + 2*alpha(1)*alpha(3)*Ival  \
    #    + 2*alpha(2)*alpha(3)*Dval + alpha(1)^2*Ival^2 \ 
    #    + 2* alpha(1)*alpha(2)*Ival*Dval + alpha(2)^2*dval^2 )) 
    
    
    #ExpWeightedFeatureMapDensity computations: 
    #accum = sum_d( \
    #  self.weights[d]*self.feature_map.get_mapped_feature_vec_element(d))
    #exponential_density = np.exp(-self.lamda*accum)*self.lamda
    
    
    l_alpha_est_non=[-0.001929, 0.010123, 3.937502]  
    r_alpha_est_non=[-0.001929, 0.010123, 3.937502]      
    left_weights_temp = [0.002149, -0.002069, 5.258745]
    left_weights = [left_weights_temp[2]*left_weights_temp[2], \
                   2*left_weights_temp[0]*left_weights_temp[2], \
                    2*left_weights_temp[1]*left_weights_temp[2], \
                    left_weights_temp[0]*left_weights_temp[0], \
                    2*left_weights_temp[0]*left_weights_temp[1], \
                    left_weights_temp[1]*left_weights_temp[1] ]
    left_lambda = 1.0
    left_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_lung_distance_map], left_weights, \
       left_polynomial_feature_map, left_lambda)
    left_likelihood = left_weighted_density.compute()
    
    l_alpha_est_right=[0.001241, -0.025153, 4.609616]

    left_weights_given_right = [l_alpha_est_right[2]*l_alpha_est_right[2], \
                    2*l_alpha_est_right[0]*l_alpha_est_right[2], \
                    2*l_alpha_est_right[1]*l_alpha_est_right[2], \
                    l_alpha_est_right[0]*l_alpha_est_right[0], \
                    2*l_alpha_est_right[0]*l_alpha_est_right[1], \
                    l_alpha_est_right[1]*l_alpha_est_right[1] ]
    left_given_right_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_lung_distance_map], 
       left_weights_given_right, left_polynomial_feature_map, left_lambda)
    LdIgivenRlung = left_given_right_weighted_density.compute()
    
    left_weights_given_nonlung = [l_alpha_est_non[2]*l_alpha_est_non[2], \
                    2*l_alpha_est_non[0]*l_alpha_est_non[2], \
                    2*l_alpha_est_non[1]*l_alpha_est_non[2], \
                    l_alpha_est_non[0]*l_alpha_est_non[0], \
                    2*l_alpha_est_non[0]*l_alpha_est_non[1], \
                    l_alpha_est_non[1]*l_alpha_est_non[1] ]
    left_given_nonlung_weighted_density = ExpWeightedFeatureMapDensity([ \
      input_image.astype(np.float),left_lung_distance_map], \
      left_weights_given_nonlung, left_polynomial_feature_map, left_lambda)
    LdIgivenNlung = left_given_nonlung_weighted_density.compute()
    
    
    right_weights_temp = [0.002149, -0.002069, 5.258745]
    right_weights = [right_weights_temp[2]*right_weights_temp[2], \
                    2*right_weights_temp[0]*right_weights_temp[2], \
                    2*right_weights_temp[1]*right_weights_temp[2], \
                    right_weights_temp[0]*right_weights_temp[0], \
                    2*right_weights_temp[0]*right_weights_temp[1], \
                    right_weights_temp[1]*right_weights_temp[1] ]
    right_lambda = 1.0
    right_weighted_density = ExpWeightedFeatureMapDensity([input_image, \
           right_lung_distance_map], right_weights, \
           right_polynomial_feature_map, right_lambda)
    right_likelihood = right_weighted_density.compute()
    
    r_alpha_est_left=[0.001241, -0.025153, 4.609616]
    right_weights_given_left = [r_alpha_est_left[2]*r_alpha_est_left[2], \
                    2*r_alpha_est_left[0]*r_alpha_est_left[2], \
                    2*r_alpha_est_left[1]*r_alpha_est_left[2], \
                    r_alpha_est_left[0]*r_alpha_est_left[0], \
                    2*r_alpha_est_left[0]*r_alpha_est_left[1], \
                    r_alpha_est_left[1]*r_alpha_est_left[1] ]
    right_lambda = 1.0
    right_given_leftlung_weighted_density = ExpWeightedFeatureMapDensity( \
          [input_image,right_lung_distance_map], right_weights_given_left, \
          right_polynomial_feature_map, right_lambda)
    RdIgivenLlung = right_given_leftlung_weighted_density.compute()
    
    right_weights_given_non = [r_alpha_est_non[2]*r_alpha_est_non[2], \
                    2*r_alpha_est_non[0]*r_alpha_est_non[2], \
                    2*r_alpha_est_non[1]*r_alpha_est_non[2], \
                    r_alpha_est_non[0]*r_alpha_est_non[0], \
                    2*r_alpha_est_non[0]*r_alpha_est_non[1], \
                    r_alpha_est_non[1]*r_alpha_est_non[1] ]
    right_lambda = 1.0
    right_given_nonlung_weighted_density = ExpWeightedFeatureMapDensity( \
         [input_image,right_lung_distance_map], right_weights_given_non, \
         right_polynomial_feature_map, right_lambda)
    RdIgivenNlung = right_given_nonlung_weighted_density.compute()

    
    notLungPrior = np.ones((length, width,np.shape( \
         probabilistic_atlases[0])[2])).astype(np.float)
    notLungPrior = notLungPrior - np.add(probabilistic_atlases[0], \
         probabilistic_atlases[1]);
    
    p_I_dleft = np.add(np.multiply(left_likelihood.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         LdIgivenRlung.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)),np.multiply(LdIgivenNlung.astype(np.float), \
         notLungPrior.astype(np.float)))  
         
         
    p_I_dright = np.add(np.multiply(RdIgivenLlung.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         right_likelihood.astype(np.float), \
         probabilistic_atlases[1].astype(np.float)),np.multiply( \
         RdIgivenNlung.astype(np.float),notLungPrior.astype(np.float)))  
    
    zero_indeces = (p_I_dleft == 0)
    p_I_dleft[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces2 = (p_I_dright == 0)
    p_I_dright[zero_indeces2] = 0.000000000000000000000001
    
    left_likelihood[tozero_indeces_intensities]=0
    right_likelihood[tozero_indeces_intensities]=0
    
    
    #save likelihoods for debugging purposes

    #segment given feature vectors
    segmented_labels = segment_chest_with_atlas([left_likelihood.astype( \
       np.float), right_likelihood.astype(np.float)], probabilistic_atlases, \
       [p_I_dleft.astype(np.float), p_I_dright.astype(np.float)])
    
    return segmented_labels
    
    
    
def compute_structure_posterior_probabilities(likelihoods, priors, \
    normalization_constants):
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
    
    normalization_constants : list of float arrays with shape (L, M, N)
        constant for each voxel and each class in order to render the output
        a true posterior probability.
        ...
        
    Returns
    -------
    energies : List of float arrays with shape (L, M, N) representing posterior 
              energies for each structure/non structure
    """
    
    #get the number of classes, initialize list of posteriors
    num_classes = np.shape(likelihoods)[0] 
    assert num_classes == np.shape(priors)[0] 
    
    # make sure none of the normalization value are = 0
    for d in range(0, num_classes):
        assert (normalization_constants[d].all() != 0)
    
    posteriors = np.zeros(np.shape(likelihoods))
    
    for d in range(0, num_classes):
        posteriors[d] = likelihoods[d]*priors[d] /(normalization_constants[d])
        
    return posteriors
    
def obtain_graph_cuts_segmentation(structure_posterior_energy, \
     not_structure_posterior_energy):
    """Obtains the graph cuts segmentation for a structure given the posterior 
       energies.  
    
    Parameters
    ----------
    structure_posterior_energy: A float array with shape (L, M, N) 
            representing the posterior energies for the structure of interest. 
            (source) 
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
    
        source_slice = structure_posterior_energy[:,:,slice_num: \
           (slice_num+1)].squeeze().astype(np.int32) 
        sink_slice = not_structure_posterior_energy[:,:,slice_num: \
           (slice_num+1)].squeeze().astype(np.int32) 

        imageIndexArray =  np.arange(numNodes).reshape(np.shape( \
           source_slice)[0], np.shape(source_slice)[1])
 
        #Adding neighbourhood terms 
        inds = np.arange(imageIndexArray.size).reshape(imageIndexArray.shape) 
        #goes from [[0,1,...numcols-1],[numcols, ...],..[.., num_elem-1]]
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()] 
        #all rows, not last col make to 1d
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()] 
        #all rows, not first col, make to 1d
        edges = np.vstack([horz, vert]).astype(np.int32) 
        #horz is first element, vert is 
        theweights = np.ones((np.shape(edges))).astype(np.int32)*18
        edges = np.hstack((edges,theweights))[:,0:3].astype(np.int32) 
        #stack the weight value hor next to edge indeces
    
        #3rd order neighbours
        horz = np.c_[inds[:, :-2].ravel(), inds[:,2:].ravel()] 
        #all rows, not last col make to 1d
        vert = np.c_[inds[:-2, :].ravel(), inds[2:, :].ravel()] 
        #all rows, not first col, make to 1d
        edges2 = np.vstack([horz, vert]).astype(np.int32) 
        #horz is first element, vert is 
        theweights2 = np.ones((np.shape(edges2))).astype(np.int32)
        edges2 = np.hstack((edges2,theweights2))[:,0:3].astype(np.int32)
    
        edges = np.vstack([edges,edges2]).astype(np.int32)

        pairwise_cost = np.array([[0, 1], [1, 0]], dtype = np.int32)
    
        energies = np.dstack((np.array(source_slice).astype(np.int32).flatten(), \
        np.array(sink_slice).astype(np.int32).flatten())).squeeze()

        segmented_slice = cut_from_graph(edges, energies, pairwise_cost, 3, \
          'expansion') 
        segmented_image[:,:,slice_num] = segmented_slice.reshape(length,width)

    return segmented_image
