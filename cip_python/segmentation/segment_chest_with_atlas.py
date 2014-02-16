import numpy as np
from scipy import ndimage
from scipy import stats
from pygco import cut_from_graph
from cip_python.utils.weighted_feature_map_densities \
    import ExpWeightedFeatureMapDensity
from cip_python.utils.feature_maps import PolynomialFeatureMap
import nrrd
import pylab as plt
import sys


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
    print("computing posterior probabilities\n")
    num_classes = np.shape(priors)[0]
    posterior_probabilities = np.zeros(np.shape(priors), dtype=np.float)
    label_map=np.zeros(np.shape(priors), dtype=np.int)
    
    posterior_probabilities = \
       compute_structure_posterior_probabilities(likelihoods, priors, \
                                                  normalization_constants)
    
    right_atlas_filename_major = "/Users/rolaharmouche/Documents/Data/COPDGene/10004O/10004O_INSP_STD_BWH_COPD/10004O_INSP_STD_BWH_COPD_leftPecMajorPost.nrrd"
    nrrd.write(right_atlas_filename_major, posterior_probabilities[0])       
    right_atlas_filename_minor = "/Users/rolaharmouche/Documents/Data/COPDGene/10004O/10004O_INSP_STD_BWH_COPD/10004O_INSP_STD_BWH_COPD_leftPecMinorPost.nrrd"
    nrrd.write(right_atlas_filename_minor, posterior_probabilities[1])       

    
    print("getting graph cuts segmentations\n")
    #  For each structure separately, input the posterior energies into
    # the graph cuts code and obtain a segmentation   
    for i in range(num_classes):
        #not_class_posterior_energies= (posterior_probabilities[i].squeeze()* \
        #  4000).astype(np.int32)
        not_class_posterior_energies= (posterior_probabilities[i]* \
          4000).astype(np.int32)
        class_posterior_energies = ( (np.ones(np.shape( \
          not_class_posterior_energies)).astype(np.float)- \
          posterior_probabilities[i].squeeze())*4000).astype(np.int32)
        label_map[i]=obtain_graph_cuts_segmentation( \
          not_class_posterior_energies, class_posterior_energies)
    
    for i in range(0, num_classes):
        label_map[i] = ndimage.binary_fill_holes(label_map[i]).astype(int)
        label_map[i] = ndimage.binary_fill_holes(label_map[i]).astype(int)   
        
           
    print("selecting label with least posterior energy")             
    #remove overlap between labels by choosing the label with the least energy
    total_labels = np.zeros([np.shape(label_map[0])[0],np.shape(label_map[0])[1],np.shape(label_map[0])[2]], dtype=int)
    #total_labels = np.zeros([np.shape(label_map[0])], dtype=int)
    isMultiple_labels = np.zeros([np.shape(label_map[0])[0],np.shape(label_map[0])[1],np.shape(label_map[0])[2]], dtype=int)
    lowest_energy_label = np.zeros([np.shape(label_map[0])[0],np.shape(label_map[0])[1],np.shape(label_map[0])[2]], dtype=int)
    for i in range(0, num_classes):
        total_labels = np.add(total_labels, label_map[i])
    isMultiple_labels = (total_labels >1)
    
    #find lowest energy label, right now maximum posterior probability
    lowest_energy_label = np.argmax(posterior_probabilities, axis=0)
    
    #isMultiple_labels&lowest or nonmultiple&current
    #or multiple & (non-lowest) = 0

        
    for i in range(0, num_classes):    
        isNotLowestEnergyLabel = (i != lowest_energy_label)       
        label_map[i][(np.logical_and(isMultiple_labels,isNotLowestEnergyLabel))] = 0 
    

    return label_map

def segment_lung_with_atlas_gaussian(input_image, probabilistic_atlases, gauss_parameters): 
    #TODO: and command line option to this command
    
    """Segment lung using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlases : list of float arrays, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of the class.
        
    gauss_parameters: Parameters of the lileihood gaussian distribution
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    length  = np.shape(probabilistic_atlases[0])[0]
    width = np.shape(probabilistic_atlases[0])[1]
    
    #Define lung area to segment
    lungPrior = probabilistic_atlases[0].astype(np.float) + probabilistic_atlases[1].astype(np.float)
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
    
    
    leftLungMean = np.array([gauss_parameters[0]])
    leftLungCovariance = np.matrix([[ gauss_parameters[1]]])
    notLungMean = np.array([ gauss_parameters[2]])
    notLungCovariance= np.matrix([[gauss_parameters[3] ]])
    rightLungMean = np.array([gauss_parameters[4]])
    rightLungCovariance = np.matrix([[gauss_parameters[5] ]])

   
    print("Calculate likelihood and posterior\n")
    left_likelihood = stats.norm.pdf(input_image.astype(np.float), loc=leftLungMean, scale=leftLungCovariance)
    notLungLikelihood = stats.norm.pdf(input_image.astype(np.float), loc=notLungMean, scale=notLungCovariance)
    right_likelihood = stats.norm.pdf(input_image.astype(np.float), loc=leftLungMean, scale=leftLungCovariance)
       
    notLungPrior = np.ones((length, width,np.shape(probabilistic_atlases[0])[2])).astype(np.float)
    notLungPrior = notLungPrior.astype(np.float) - (np.add(probabilistic_atlases[0].astype(np.float), probabilistic_atlases[1].astype(np.float)));
    
    cap_not_lung = (notLungPrior<0)
    notLungPrior[cap_not_lung]=0
    
    ones_indeces_notprior= notLungPrior > 100.0
    notLungPrior[ones_indeces_notprior]=100
       
    p_I_dleft = np.add(np.multiply(left_likelihood.astype(np.float), \
         lungPrior.astype( \
         np.float)),np.multiply(notLungLikelihood.astype(np.float), \
         notLungPrior.astype(np.float)))  
         
    p_I_dright = np.add(np.multiply(left_likelihood.astype(np.float), \
         lungPrior.astype( \
         np.float)),np.multiply(notLungLikelihood.astype(np.float), \
         notLungPrior.astype(np.float)))  
         
    zero_indeces = (p_I_dleft == 0)
    p_I_dleft[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces2 = (p_I_dright == 0)
    p_I_dright[zero_indeces2] = 0.000000000000000000000001
    
    left_likelihood[tozero_indeces_intensities]=0
    right_likelihood[tozero_indeces_intensities]=0
    
    #debugging purposes
    leftLungPosterior = np.divide(np.multiply(probabilistic_atlases[0].astype(np.float),left_likelihood.astype(np.float)),p_I_dleft.astype(np.float))  
    rightLungPosterior = np.divide(np.multiply(probabilistic_atlases[1].astype(np.float),right_likelihood.astype(np.float)),p_I_dleft.astype(np.float))  
    notLungPosterior = np.multiply(notLungLikelihood.astype(np.float),notLungPrior.astype(np.float))   
    
    #segment given feature vectors
    segmented_labels = segment_chest_with_atlas([left_likelihood.astype( \
       np.float), left_likelihood.astype(np.float)], probabilistic_atlases, \
       [p_I_dleft.astype(np.float), p_I_dright.astype(np.float)])
    
    return segmented_labels
    
def segment_lung_with_atlas(input_image, probabilistic_atlases, exponential_parameters): 
    #TODO: and command line option to this command
    
    """Segment lung using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlases : list of float arrays, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of the class.
        
    exponential_parameters: Parameters of the exponential likelihood distribution
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    #compute feature vectors for left and right lungs
    ### TODO: replace with likelihood params for class 0.


    length  = np.shape(probabilistic_atlases[0])[0]
    width = np.shape(probabilistic_atlases[0])[1]
    
    #Define lung area to segment
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
    
    
    left_lung_distance_map = compute_distance_to_atlas(probabilistic_atlases[0])   
    right_lung_distance_map = compute_distance_to_atlas(probabilistic_atlases[1])
       
        
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
    
    #older weights
    #left_weights_temp = [0.002149, -0.002069, 5.258745] 
    #l_alpha_est_right=[0.001241, -0.025153, 4.609616]      
    #l_alpha_est_non=[-0.001929, 0.010123, 3.937502]  
    #
    #right_weights_temp = [0.002149, -0.002069, 5.258745] 
    #r_alpha_est_left=[0.001241, -0.025153, 4.609616]      
    #r_alpha_est_non=[-0.001929, 0.010123, 3.937502]  
        
    #r_alpha_est_left=[0.002500, -0.095894, 6.786622]
    #right_weights_temp=[0.001245, 0.066628, 4.269774]
    #r_alpha_est_non=[-0.001433, -0.005590, 4.143140]
    
    
    #newer 

#    left_weights_temp=[0.002242, 0.002139, 5.305966]
#    l_alpha_est_right=[0.001987, -0.054164, 5.415881]
#    l_alpha_est_non=[-0.001288, -0.034694, 4.225687]
#
#    r_alpha_est_left=[0.002209, -0.094936, 6.731629]
#    right_weights_temp=[0.001689, 0.072709, 4.398574]
#    r_alpha_est_non=[-0.000816, -0.035418, 4.499488]
    

    #newer, 80 bins instead of 50: left works, right doesnt
    #left_weights_temp=[0.002312, 0.002666, 5.806209]
    #l_alpha_est_right=[0.001729, -0.029253, 5.383404]
    #l_alpha_est_non=[-0.001127, 0.009051, 4.617099]
    #
    #r_alpha_est_left=[0.001914, -0.060901, 6.623247]
    #right_weights_temp=[0.001878, 0.073107, 5.053620]
    #r_alpha_est_non=[-0.000779, -0.029794, 5.143489]




    #l_alpha_est_right=[0.001816, 0.001857, 5.921779]
    #left_weights_temp=[0.002463, 0.118871, 5.827013]
    #l_alpha_est_non=[-0.000091, 0.012917, 4.446526]
    #right_weights_temp=[0.001594, 0.030955, 5.626165] #this is the source of the problemmmm
    #r_alpha_est_left=[0.001946, 0.009239, 5.685427]
    #r_alpha_est_non=[-0.000090, 0.014221, 4.432606]

    #below working
    #alpha_dleft_given_left = [0.002149, -0.002069, 5.258745] 
    #alpha_dleft_given_right=[0.001241, -0.025153, 4.609616]      
    #alpha_dleft_given_non=[-0.001929, 0.010123, 3.937502]  
    ##
    #alpha_dright_given_right = [0.002149, -0.002069, 5.258745] 
    #alpha_dright_given_left=[0.001241, -0.025153, 4.609616]      
    #alpha_dright_given_non=[-0.001929, 0.010123, 3.937502]  
        
    alpha_dleft_given_left = exponential_parameters[0]
    alpha_dleft_given_right=exponential_parameters[1]      
    alpha_dleft_given_non=exponential_parameters[2] 
    #
    alpha_dright_given_right = exponential_parameters[3]
    alpha_dright_given_left=exponential_parameters[4]     
    alpha_dright_given_non=exponential_parameters[5]  


    left_weights = [alpha_dleft_given_left[2]*alpha_dleft_given_left[2], \
                   2*alpha_dleft_given_left[0]*alpha_dleft_given_left[2], \
                    2*alpha_dleft_given_left[1]*alpha_dleft_given_left[2], \
                    alpha_dleft_given_left[0]*alpha_dleft_given_left[0], \
                    2*alpha_dleft_given_left[0]*alpha_dleft_given_left[1], \
                    alpha_dleft_given_left[1]*alpha_dleft_given_left[1] ]
    left_lambda = 1.0
    left_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_lung_distance_map], left_weights, \
       left_polynomial_feature_map, left_lambda)
    left_likelihood = left_weighted_density.compute()
    
    

    left_weights_given_right = [alpha_dleft_given_right[2]*alpha_dleft_given_right[2], \
                    2*alpha_dleft_given_right[0]*alpha_dleft_given_right[2], \
                    2*alpha_dleft_given_right[1]*alpha_dleft_given_right[2], \
                    alpha_dleft_given_right[0]*alpha_dleft_given_right[0], \
                    2*alpha_dleft_given_right[0]*alpha_dleft_given_right[1], \
                    alpha_dleft_given_right[1]*alpha_dleft_given_right[1] ]
    left_given_right_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_lung_distance_map], 
       left_weights_given_right, left_polynomial_feature_map, left_lambda)
    LdIgivenRlung = left_given_right_weighted_density.compute()
    
    left_weights_given_nonlung = [alpha_dleft_given_non[2]*alpha_dleft_given_non[2], \
                    2*alpha_dleft_given_non[0]*alpha_dleft_given_non[2], \
                    2*alpha_dleft_given_non[1]*alpha_dleft_given_non[2], \
                    alpha_dleft_given_non[0]*alpha_dleft_given_non[0], \
                    2*alpha_dleft_given_non[0]*alpha_dleft_given_non[1], \
                    alpha_dleft_given_non[1]*alpha_dleft_given_non[1] ]
    left_given_nonlung_weighted_density = ExpWeightedFeatureMapDensity([ \
      input_image.astype(np.float),left_lung_distance_map], \
      left_weights_given_nonlung, left_polynomial_feature_map, left_lambda)
    LdIgivenNlung = left_given_nonlung_weighted_density.compute()
    
    
    
    right_weights = [alpha_dright_given_right[2]*alpha_dright_given_right[2], \
                    2*alpha_dright_given_right[0]*alpha_dright_given_right[2], \
                    2*alpha_dright_given_right[1]*alpha_dright_given_right[2], \
                    alpha_dright_given_right[0]*alpha_dright_given_right[0], \
                    2*alpha_dright_given_right[0]*alpha_dright_given_right[1], \
                    alpha_dright_given_right[1]*alpha_dright_given_right[1] ]
    right_lambda = 1.0
    right_weighted_density = ExpWeightedFeatureMapDensity([input_image, \
           right_lung_distance_map], right_weights, \
           right_polynomial_feature_map, right_lambda)
    right_likelihood = right_weighted_density.compute()
    
    
    right_weights_given_left = [alpha_dright_given_left[2]*alpha_dright_given_left[2], \
                    2*alpha_dright_given_left[0]*alpha_dright_given_left[2], \
                    2*alpha_dright_given_left[1]*alpha_dright_given_left[2], \
                    alpha_dright_given_left[0]*alpha_dright_given_left[0], \
                    2*alpha_dright_given_left[0]*alpha_dright_given_left[1], \
                    alpha_dright_given_left[1]*alpha_dright_given_left[1] ]
    right_lambda = 1.0
    right_given_leftlung_weighted_density = ExpWeightedFeatureMapDensity( \
          [input_image,right_lung_distance_map], right_weights_given_left, \
          right_polynomial_feature_map, right_lambda)
    RdIgivenLlung = right_given_leftlung_weighted_density.compute()
    
    right_weights_given_non = [alpha_dright_given_non[2]*alpha_dright_given_non[2], \
                    2*alpha_dright_given_non[0]*alpha_dright_given_non[2], \
                    2*alpha_dright_given_non[1]*alpha_dright_given_non[2], \
                    alpha_dright_given_non[0]*alpha_dright_given_non[0], \
                    2*alpha_dright_given_non[0]*alpha_dright_given_non[1], \
                    alpha_dright_given_non[1]*alpha_dright_given_non[1] ]
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
    

    #segment given feature vectors
    segmented_labels = segment_chest_with_atlas([left_likelihood.astype( \
       np.float), right_likelihood.astype(np.float)], probabilistic_atlases, \
       [p_I_dleft.astype(np.float), p_I_dright.astype(np.float)])
    
    return segmented_labels
    

def segment_pec_with_atlas(input_image, probabilistic_atlases, exponential_parameters): 
    #TODO: and command line option to this command
    
    """Segment pecs using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlases : list of float arrays, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of the class.
        
    exponential_parameters: Parameters of the exponential likelihood distribution
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    #compute feature vectors for left and right lungs
    ### TODO: replace with likelihood params for class 0.


    length  = np.shape(probabilistic_atlases[0])[0]
    width = np.shape(probabilistic_atlases[0])[1]
    
    #Define pec area to segment
    pecPrior = probabilistic_atlases[0] + probabilistic_atlases[1] +\
       + probabilistic_atlases[2]+ probabilistic_atlases[3]
    zero_indeces_thresholding = pecPrior < 0.25 
    pecPriorSlicedialated = pecPrior
    pecPriorSlicedialated[zero_indeces_thresholding] = 0.0
    
    ones_indeces_thresholding = pecPrior > 0.24 
    pecPriorSlicedialated[ones_indeces_thresholding] = 1.0

    pecPriorSlicedialated = ndimage.binary_dilation(pecPriorSlicedialated, \
      iterations=8)
    ndimage.morphology.binary_fill_holes(pecPriorSlicedialated, \
      structure=None, output=pecPriorSlicedialated, origin=0)
    tozero_indeces_intensities = pecPriorSlicedialated < 1
           
    left_major_distance_map = compute_distance_to_atlas(probabilistic_atlases[0].astype(float))
    left_minor_distance_map = compute_distance_to_atlas(probabilistic_atlases[1].astype(float))     
    right_major_distance_map = compute_distance_to_atlas(probabilistic_atlases[2].astype(float))
    right_minor_distance_map = compute_distance_to_atlas(probabilistic_atlases[3].astype(float))
 
         
    left_major_polynomial_feature_map = PolynomialFeatureMap( [input_image, \
      left_major_distance_map],[0,1,2] )  
    left_major_polynomial_feature_map.compute_num_terms()
    
    right_major_polynomial_feature_map = PolynomialFeatureMap( [input_image, \
      right_major_distance_map],[0,1,2] )  
    right_major_polynomial_feature_map.compute_num_terms()


    left_minor_polynomial_feature_map = PolynomialFeatureMap( [input_image, \
      left_minor_distance_map],[0,1,2] )  
    left_minor_polynomial_feature_map.compute_num_terms()
    
    right_minor_polynomial_feature_map = PolynomialFeatureMap( [input_image, \
      right_minor_distance_map],[0,1,2] )  
    right_minor_polynomial_feature_map.compute_num_terms()
    
    
    #define the weights
    print("defining the weights, atlas shape is:")     
    print(np.shape(probabilistic_atlases))
    print(np.shape(input_image))
    
    alpha_dleftmajor_given_leftmajor=exponential_parameters[0]
    alpha_dleftminor_given_leftmajor=exponential_parameters[1]
    alpha_drightmajor_given_leftmajor=exponential_parameters[2]
    alpha_drightminor_given_leftmajor=exponential_parameters[3]
    alpha_dleftmajor_given_leftminor=exponential_parameters[4]
    alpha_dleftminor_given_leftminor=exponential_parameters[5]
    alpha_drightmajor_given_leftminor=exponential_parameters[6]
    alpha_drightminor_given_leftminor=exponential_parameters[7]
    alpha_dleftmajor_given_rightmajor=exponential_parameters[8]
    alpha_dleftminor_given_rightmajor=exponential_parameters[9]
    alpha_drightmajor_given_rightmajor=exponential_parameters[10]
    alpha_drightminor_given_rightmajor=exponential_parameters[11]
    alpha_dleftmajor_given_rightminor=exponential_parameters[12]
    alpha_dleftminor_given_rightminor=exponential_parameters[13]
    alpha_drightmajor_given_rightminor=exponential_parameters[14]
    alpha_drightminor_given_rightminor=exponential_parameters[15]
    alpha_dleftmajor_given_non=exponential_parameters[16]
    alpha_dleftminor_given_non=exponential_parameters[17]
    alpha_drightmajor_given_non=exponential_parameters[18]
    alpha_drightminor_given_non=exponential_parameters[19]
    #left major
    
    
    ##something wrongwith this first one
    left_major_weights = [alpha_dleftmajor_given_leftmajor[2]*alpha_dleftmajor_given_leftmajor[2], \
                   2*alpha_dleftmajor_given_leftmajor[0]*alpha_dleftmajor_given_leftmajor[2], \
                    2*alpha_dleftmajor_given_leftmajor[1]*alpha_dleftmajor_given_leftmajor[2], \
                    alpha_dleftmajor_given_leftmajor[0]*alpha_dleftmajor_given_leftmajor[0], \
                    2*alpha_dleftmajor_given_leftmajor[0]*alpha_dleftmajor_given_leftmajor[1], \
                    alpha_dleftmajor_given_leftmajor[1]*alpha_dleftmajor_given_leftmajor[1] ]
    left_major_lambda = 1.0
    left_major_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_major_distance_map], left_major_weights, \
       left_major_polynomial_feature_map, left_major_lambda)
    left_majorlikelihood = left_major_weighted_density.compute()
    
    left_major_weights_given_left_minor  = [alpha_dleftmajor_given_leftminor[2]*\
                   alpha_dleftmajor_given_leftminor[2], \
                   2*alpha_dleftmajor_given_leftminor[0]*alpha_dleftmajor_given_leftminor[2], \
                    2*alpha_dleftmajor_given_leftminor[1]*alpha_dleftmajor_given_leftminor[2], \
                    alpha_dleftmajor_given_leftminor[0]*alpha_dleftmajor_given_leftminor[0], \
                    2*alpha_dleftmajor_given_leftminor[0]*alpha_dleftmajor_given_leftminor[1], \
                    alpha_dleftmajor_given_leftminor[1]*alpha_dleftmajor_given_leftminor[1] ]
    left_major_lambda = 1.0
    left_major_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_major_distance_map], left_major_weights_given_left_minor, \
       left_major_polynomial_feature_map, left_major_lambda)
    leftMajorIgivenLeftMinor = left_major_given_leftminor_weighted_density.compute()
    
    print("conditional shapes are") 
    print(np.shape(leftMajorIgivenLeftMinor))   
    print(np.shape(left_majorlikelihood))     
    print(left_major_weights)
    left_major_weights_given_right_major = [alpha_dleftmajor_given_rightmajor[2]*\
                    alpha_dleftmajor_given_rightmajor[2], \
                    2*alpha_dleftmajor_given_rightmajor[0]*alpha_dleftmajor_given_rightmajor[2], \
                    2*alpha_dleftmajor_given_rightmajor[1]*alpha_dleftmajor_given_rightmajor[2], \
                    alpha_dleftmajor_given_rightmajor[0]*alpha_dleftmajor_given_rightmajor[0], \
                    2*alpha_dleftmajor_given_rightmajor[0]*alpha_dleftmajor_given_rightmajor[1], \
                    alpha_dleftmajor_given_rightmajor[1]*alpha_dleftmajor_given_rightmajor[1] ]
    leftmajor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_major_distance_map], 
       left_major_weights_given_right_major, left_major_polynomial_feature_map, left_major_lambda)
    leftMajorIgivenRightMajor = leftmajor_given_rightmajor_weighted_density.compute()
    
    left_major_weights_given_right_minor = [alpha_dleftmajor_given_rightminor[2]*\
                    alpha_dleftmajor_given_rightminor[2], \
                    2*alpha_dleftmajor_given_rightminor[0]*alpha_dleftmajor_given_rightminor[2], \
                    2*alpha_dleftmajor_given_rightminor[1]*alpha_dleftmajor_given_rightminor[2], \
                    alpha_dleftmajor_given_rightminor[0]*alpha_dleftmajor_given_rightminor[0], \
                    2*alpha_dleftmajor_given_rightminor[0]*alpha_dleftmajor_given_rightminor[1], \
                    alpha_dleftmajor_given_rightminor[1]*alpha_dleftmajor_given_rightminor[1] ]
    leftmajor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_major_distance_map], 
       left_major_weights_given_right_minor, left_major_polynomial_feature_map, left_major_lambda)
    leftMajorIgivenRightMinor = leftmajor_given_rightminor_weighted_density.compute()
    
    left_major_weights_given_non = [alpha_dleftmajor_given_non[2]*\
                    alpha_dleftmajor_given_non[2], \
                    2*alpha_dleftmajor_given_non[0]*alpha_dleftmajor_given_non[2], \
                    2*alpha_dleftmajor_given_non[1]*alpha_dleftmajor_given_non[2], \
                    alpha_dleftmajor_given_non[0]*alpha_dleftmajor_given_non[0], \
                    2*alpha_dleftmajor_given_non[0]*alpha_dleftmajor_given_non[1], \
                    alpha_dleftmajor_given_non[1]*alpha_dleftmajor_given_non[1] ]
    leftmajor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_major_distance_map], 
       left_major_weights_given_non, left_major_polynomial_feature_map, left_major_lambda)
    leftMajorIgivenNon = leftmajor_given_non_weighted_density.compute()
    
    #left minor
    left_minor_weights_given_left_major = [alpha_dleftminor_given_leftmajor[2]*alpha_dleftminor_given_leftmajor[2], \
                   2*alpha_dleftminor_given_leftmajor[0]*alpha_dleftminor_given_leftmajor[2], \
                    2*alpha_dleftminor_given_leftmajor[1]*alpha_dleftminor_given_leftmajor[2], \
                    alpha_dleftminor_given_leftmajor[0]*alpha_dleftminor_given_leftmajor[0], \
                    2*alpha_dleftminor_given_leftmajor[0]*alpha_dleftminor_given_leftmajor[1], \
                    alpha_dleftminor_given_leftmajor[1]*alpha_dleftminor_given_leftmajor[1] ]
    left_minor_lambda = 1.0
    left_minor_given_left_major_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_minor_distance_map], left_minor_weights_given_left_major, \
       left_minor_polynomial_feature_map, left_minor_lambda)
    leftMinorIgivenLeftMajor = left_minor_given_left_major_weighted_density.compute()
    
    
    left_minor_weights = [alpha_dleftminor_given_leftminor[2]*\
                   alpha_dleftminor_given_leftminor[2], \
                   2*alpha_dleftminor_given_leftminor[0]*alpha_dleftminor_given_leftminor[2], \
                    2*alpha_dleftminor_given_leftminor[1]*alpha_dleftminor_given_leftminor[2], \
                    alpha_dleftminor_given_leftminor[0]*alpha_dleftminor_given_leftminor[0], \
                    2*alpha_dleftminor_given_leftminor[0]*alpha_dleftminor_given_leftminor[1], \
                    alpha_dleftminor_given_leftminor[1]*alpha_dleftminor_given_leftminor[1] ]
    left_minor_lambda = 1.0
    left_minor_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_minor_distance_map], left_minor_weights, \
       left_minor_polynomial_feature_map, left_minor_lambda)
    left_minorlikelihood = left_minor_given_leftminor_weighted_density.compute()
        

    left_minor_weights_given_right_major = [alpha_dleftminor_given_rightmajor[2]*\
                    alpha_dleftminor_given_rightmajor[2], \
                    2*alpha_dleftminor_given_rightmajor[0]*alpha_dleftminor_given_rightmajor[2], \
                    2*alpha_dleftminor_given_rightmajor[1]*alpha_dleftminor_given_rightmajor[2], \
                    alpha_dleftminor_given_rightmajor[0]*alpha_dleftminor_given_rightmajor[0], \
                    2*alpha_dleftminor_given_rightmajor[0]*alpha_dleftminor_given_rightmajor[1], \
                    alpha_dleftminor_given_rightmajor[1]*alpha_dleftminor_given_rightmajor[1] ]
    leftminor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_minor_distance_map], 
       left_minor_weights_given_right_major, left_minor_polynomial_feature_map, left_minor_lambda)
    leftMinorIgivenRightMajor = leftminor_given_rightmajor_weighted_density.compute()
    
    left_minor_weights_given_right_minor = [alpha_dleftminor_given_rightminor[2]*\
                    alpha_dleftminor_given_rightminor[2], \
                    2*alpha_dleftminor_given_rightminor[0]*alpha_dleftminor_given_rightminor[2], \
                    2*alpha_dleftminor_given_rightminor[1]*alpha_dleftminor_given_rightminor[2], \
                    alpha_dleftminor_given_rightminor[0]*alpha_dleftminor_given_rightminor[0], \
                    2*alpha_dleftminor_given_rightminor[0]*alpha_dleftminor_given_rightminor[1], \
                    alpha_dleftminor_given_rightminor[1]*alpha_dleftminor_given_rightminor[1] ]
    leftminor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_minor_distance_map], 
       left_minor_weights_given_right_minor, left_minor_polynomial_feature_map, left_minor_lambda)
    leftMinorIgivenRightMinor = leftminor_given_rightminor_weighted_density.compute()

    left_minor_weights_given_non = [alpha_dleftminor_given_non[2]*\
                    alpha_dleftminor_given_non[2], \
                    2*alpha_dleftminor_given_non[0]*alpha_dleftminor_given_non[2], \
                    2*alpha_dleftminor_given_non[1]*alpha_dleftminor_given_non[2], \
                    alpha_dleftminor_given_non[0]*alpha_dleftminor_given_non[0], \
                    2*alpha_dleftminor_given_non[0]*alpha_dleftminor_given_non[1], \
                    alpha_dleftminor_given_non[1]*alpha_dleftminor_given_non[1] ]
    leftminor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_minor_distance_map], 
       left_minor_weights_given_non, left_minor_polynomial_feature_map, left_minor_lambda)
    leftMinorIgivenNon = leftminor_given_non_weighted_density.compute()          

    #right major
    
    right_major_weights_given_left_major = [alpha_drightmajor_given_leftmajor[2]*alpha_drightmajor_given_leftmajor[2], \
                   2*alpha_drightmajor_given_leftmajor[0]*alpha_drightmajor_given_leftmajor[2], \
                    2*alpha_drightmajor_given_leftmajor[1]*alpha_drightmajor_given_leftmajor[2], \
                    alpha_drightmajor_given_leftmajor[0]*alpha_drightmajor_given_leftmajor[0], \
                    2*alpha_drightmajor_given_leftmajor[0]*alpha_drightmajor_given_leftmajor[1], \
                    alpha_drightmajor_given_leftmajor[1]*alpha_drightmajor_given_leftmajor[1] ]
    right_major_lambda = 1.0
    rightmajor_given_leftmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_major_distance_map], right_major_weights_given_left_major, \
       right_major_polynomial_feature_map, right_major_lambda)
    rightMajorIgivenLeftMajor = rightmajor_given_leftmajor_weighted_density.compute()
    
    right_major_weights_given_left_minor  = [alpha_drightmajor_given_leftminor[2]*\
                   alpha_drightmajor_given_leftminor[2], \
                   2*alpha_drightmajor_given_leftminor[0]*alpha_drightmajor_given_leftminor[2], \
                    2*alpha_drightmajor_given_leftminor[1]*alpha_drightmajor_given_leftminor[2], \
                    alpha_drightmajor_given_leftminor[0]*alpha_drightmajor_given_leftminor[0], \
                    2*alpha_drightmajor_given_leftminor[0]*alpha_drightmajor_given_leftminor[1], \
                    alpha_drightmajor_given_leftminor[1]*alpha_drightmajor_given_leftminor[1] ]
    right_major_lambda = 1.0
    right_major_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_major_distance_map], right_major_weights_given_left_minor, \
       left_major_polynomial_feature_map, right_major_lambda)
    rightMajorIgivenLeftMinor = right_major_given_leftminor_weighted_density.compute()
    
          

    right_major_weights = [alpha_drightmajor_given_rightmajor[2]*\
                    alpha_drightmajor_given_rightmajor[2], \
                    2*alpha_drightmajor_given_rightmajor[0]*alpha_dleftmajor_given_rightmajor[2], \
                    2*alpha_drightmajor_given_rightmajor[1]*alpha_drightmajor_given_rightmajor[2], \
                    alpha_drightmajor_given_rightmajor[0]*alpha_drightmajor_given_rightmajor[0], \
                    2*alpha_drightmajor_given_rightmajor[0]*alpha_drightmajor_given_rightmajor[1], \
                    alpha_drightmajor_given_rightmajor[1]*alpha_drightmajor_given_rightmajor[1] ]
    rightmajor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_major_distance_map], 
       right_major_weights, right_major_polynomial_feature_map, right_major_lambda)
    right_majorlikelihood = rightmajor_given_rightmajor_weighted_density.compute()
    
    
    right_major_weights_given_right_minor = [alpha_drightmajor_given_rightminor[2]*\
                    alpha_drightmajor_given_rightminor[2], \
                    2*alpha_drightmajor_given_rightminor[0]*alpha_drightmajor_given_rightminor[2], \
                    2*alpha_drightmajor_given_rightminor[1]*alpha_drightmajor_given_rightminor[2], \
                    alpha_drightmajor_given_rightminor[0]*alpha_drightmajor_given_rightminor[0], \
                    2*alpha_drightmajor_given_rightminor[0]*alpha_drightmajor_given_rightminor[1], \
                    alpha_drightmajor_given_rightminor[1]*alpha_drightmajor_given_rightminor[1] ]
    rightmajor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_major_distance_map], 
       right_major_weights_given_right_minor, right_major_polynomial_feature_map, right_major_lambda)
    rightMajorIgivenRightMinor = rightmajor_given_rightminor_weighted_density.compute()
    
    right_major_weights_given_non = [alpha_drightmajor_given_non[2]*\
                    alpha_drightmajor_given_non[2], \
                    2*alpha_drightmajor_given_non[0]*alpha_drightmajor_given_non[2], \
                    2*alpha_drightmajor_given_non[1]*alpha_drightmajor_given_non[2], \
                    alpha_drightmajor_given_non[0]*alpha_drightmajor_given_non[0], \
                    2*alpha_drightmajor_given_non[0]*alpha_drightmajor_given_non[1], \
                    alpha_drightmajor_given_non[1]*alpha_drightmajor_given_non[1] ]
    rightmajor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_major_distance_map], 
       right_major_weights_given_non, right_major_polynomial_feature_map, right_major_lambda)
    rightMajorIgivenNon = rightmajor_given_non_weighted_density.compute()
        
   #right minor
    right_minor_weights_given_left_major = [alpha_drightminor_given_leftmajor[2]*alpha_drightminor_given_leftmajor[2], \
                   2*alpha_drightminor_given_leftmajor[0]*alpha_drightminor_given_leftmajor[2], \
                    2*alpha_drightminor_given_leftmajor[1]*alpha_drightminor_given_leftmajor[2], \
                    alpha_drightminor_given_leftmajor[0]*alpha_drightminor_given_leftmajor[0], \
                    2*alpha_drightminor_given_leftmajor[0]*alpha_drightminor_given_leftmajor[1], \
                    alpha_drightminor_given_leftmajor[1]*alpha_drightminor_given_leftmajor[1] ]
    right_minor_lambda = 1.0
    right_minor_given_left_major_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_minor_distance_map], right_minor_weights_given_left_major, \
       right_minor_polynomial_feature_map, right_minor_lambda)
    rightMinorIgivenLeftMajor = right_minor_given_left_major_weighted_density.compute()
    
 
    right_minor_weights_given_left_minor = [alpha_drightminor_given_leftminor[2]*\
                   alpha_drightminor_given_leftminor[2], \
                   2*alpha_drightminor_given_leftminor[0]*alpha_drightminor_given_leftminor[2], \
                    2*alpha_drightminor_given_leftminor[1]*alpha_drightminor_given_leftminor[2], \
                    alpha_drightminor_given_leftminor[0]*alpha_drightminor_given_leftminor[0], \
                    2*alpha_drightminor_given_leftminor[0]*alpha_drightminor_given_leftminor[1], \
                    alpha_drightminor_given_leftminor[1]*alpha_drightminor_given_leftminor[1] ]
    right_minor_lambda = 1.0
    right_minor_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),left_minor_distance_map], right_minor_weights_given_left_minor, \
       right_minor_polynomial_feature_map, right_minor_lambda)
    rightMinorIgivenLeftMinor = right_minor_given_leftminor_weighted_density.compute()
        

    right_minor_weights_given_right_major = [alpha_drightminor_given_rightmajor[2]*\
                    alpha_drightminor_given_rightmajor[2], \
                    2*alpha_drightminor_given_rightmajor[0]*alpha_drightminor_given_rightmajor[2], \
                    2*alpha_drightminor_given_rightmajor[1]*alpha_drightminor_given_rightmajor[2], \
                    alpha_drightminor_given_rightmajor[0]*alpha_drightminor_given_rightmajor[0], \
                    2*alpha_drightminor_given_rightmajor[0]*alpha_drightminor_given_rightmajor[1], \
                    alpha_drightminor_given_rightmajor[1]*alpha_drightminor_given_rightmajor[1] ]
    rightminor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_minor_distance_map], 
       right_minor_weights_given_right_major, right_minor_polynomial_feature_map, right_minor_lambda)
    rightMinorIgivenRightMajor = rightminor_given_rightmajor_weighted_density.compute()
    
    right_minor_weights_given_right_minor = [alpha_drightminor_given_rightminor[2]*\
                    alpha_drightminor_given_rightminor[2], \
                    2*alpha_drightminor_given_rightminor[0]*alpha_drightminor_given_rightminor[2], \
                    2*alpha_drightminor_given_rightminor[1]*alpha_drightminor_given_rightminor[2], \
                    alpha_drightminor_given_rightminor[0]*alpha_drightminor_given_rightminor[0], \
                    2*alpha_drightminor_given_rightminor[0]*alpha_drightminor_given_rightminor[1], \
                    alpha_drightminor_given_rightminor[1]*alpha_drightminor_given_rightminor[1] ]
    rightminor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_minor_distance_map], 
       left_minor_weights_given_right_minor, right_minor_polynomial_feature_map, right_minor_lambda)
    right_minorlikelihood = rightminor_given_rightminor_weighted_density.compute()
  
    right_minor_weights_given_non = [alpha_drightminor_given_non[2]*\
                    alpha_drightminor_given_non[2], \
                    2*alpha_drightminor_given_non[0]*alpha_drightminor_given_non[2], \
                    2*alpha_drightminor_given_non[1]*alpha_drightminor_given_non[2], \
                    alpha_drightminor_given_non[0]*alpha_drightminor_given_non[0], \
                    2*alpha_drightminor_given_non[0]*alpha_drightminor_given_non[1], \
                    alpha_drightminor_given_non[1]*alpha_drightminor_given_non[1] ]
    rightminor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       input_image.astype(np.float),right_minor_distance_map], 
       right_minor_weights_given_non, right_minor_polynomial_feature_map, right_minor_lambda)
    rightMinorIgivenNon = rightminor_given_non_weighted_density.compute()
   
    print("computing probabilities")
    lungPrior = np.ones((length, width,1)).astype(np.float)
    lungPrior = np.add(probabilistic_atlases[0],probabilistic_atlases[1])
    lungPrior = np.add(lungPrior,probabilistic_atlases[2])
    lungPrior = np.add(lungPrior,probabilistic_atlases[3])
    
    notLungPrior = np.ones((length, width,1)).astype(np.float)
    notLungPrior = notLungPrior - lungPrior;
    
    print(np.shape(probabilistic_atlases[0]))
    print(np.shape(left_majorlikelihood))
    print(np.shape(leftMajorIgivenLeftMinor))
    print(np.shape(leftMajorIgivenRightMinor))
    print(np.shape(leftMajorIgivenRightMinor))
    print(np.shape(leftMajorIgivenNon))
    print("PI")    
    p_I_dleftmajor = np.add(
         np.multiply(left_majorlikelihood.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),
         np.multiply( leftMajorIgivenLeftMinor.astype(np.float),\
         probabilistic_atlases[1].astype(np.float)))
    print(np.shape(p_I_dleftmajor))
    p_I_dleftmajor = np.add(p_I_dleftmajor,              
         np.multiply( leftMajorIgivenRightMajor.astype(np.float),
         probabilistic_atlases[2].astype(np.float)))
    print(np.shape(p_I_dleftmajor))
    p_I_dleftmajor = np.add(p_I_dleftmajor,                   
         np.multiply( leftMajorIgivenRightMinor.astype(np.float),\
         probabilistic_atlases[3].astype(np.float)))  
    print(np.shape(p_I_dleftmajor))       
    p_I_dleftmajor = np.add(p_I_dleftmajor, 
         np.multiply(leftMajorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float))
         )  

    print(np.shape(p_I_dleftmajor))
    p_I_dleftminor = np.add(np.multiply(leftMinorIgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         left_minorlikelihood.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))
    p_I_dleftminor = np.add(p_I_dleftminor, np.multiply( \
         leftMinorIgivenRightMajor.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))      
    p_I_dleftminor = np.add(p_I_dleftminor,      np.multiply( \
         leftMinorIgivenRightMinor.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))
    p_I_dleftminor = np.add(p_I_dleftminor, np.multiply(leftMinorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))           
               
    p_I_drightmajor = np.add(np.multiply(rightMajorIgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         rightMajorIgivenLeftMinor.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))         
    p_I_drightmajor = np.add(p_I_drightmajor,     np.multiply( \
         right_majorlikelihood.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))
         
    p_I_drightmajor = np.add(p_I_drightmajor,      np.multiply( \
         rightMajorIgivenRightMinor.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))
         
    p_I_drightmajor = np.add(p_I_drightmajor,np.multiply(rightMajorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))                      
                                   
    p_I_drightminor = np.add(np.multiply(rightMinorIgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         rightMinorIgivenLeftMinor.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))
    p_I_drightminor = np.add(p_I_drightminor,     
         np.multiply( \
         leftMinorIgivenRightMinor.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))
    p_I_drightminor = np.add(p_I_drightminor, np.multiply( \
         right_minorlikelihood.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))
         
    p_I_drightminor = np.add(p_I_drightminor, np.multiply(rightMajorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))             
    
    zero_indeces = (p_I_dleftmajor == 0)
    p_I_dleftmajor[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces = (p_I_dleftminor == 0)
    p_I_dleftminor[zero_indeces] = 0.000000000000000000000001
    
    zero_indeces = (p_I_drightmajor == 0)
    p_I_drightmajor[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces = (p_I_drightminor == 0)
    p_I_drightminor[zero_indeces] = 0.000000000000000000000001
    
    left_majorlikelihood[tozero_indeces_intensities]=0
    right_majorlikelihood[tozero_indeces_intensities]=0
    left_minorlikelihood[tozero_indeces_intensities]=0
    right_minorlikelihood[tozero_indeces_intensities]=0
   
   
    right_atlas_filename_major = "/Users/rolaharmouche/Documents/Data/COPDGene/10004O/10004O_INSP_STD_BWH_COPD/10004O_INSP_STD_BWH_COPD_leftPecMajorlikelihood.nrrd"
    nrrd.write(right_atlas_filename_major, left_majorlikelihood)       

    print("array shapes are:")
    print(np.shape(left_majorlikelihood))
    print(np.shape(p_I_dleftminor))
    print(np.shape(probabilistic_atlases))
    #segment given feature vectors
    segmented_labels = segment_chest_with_atlas([left_majorlikelihood.astype( \
       np.float), left_minorlikelihood.astype(np.float), right_majorlikelihood.astype(np.float), right_minorlikelihood.astype(np.float)], probabilistic_atlases, \
       [p_I_dleftmajor.astype(np.float), p_I_dleftminor.astype(np.float),p_I_drightmajor.astype(np.float), p_I_drightminor.astype(np.float)])
    
    return segmented_labels
    
def segment_pec_with_atlas_expgaus(input_image, probabilistic_atlases, expgaus_parameters): 
    #TODO: and command line option to this command
    
    """Segment pecs using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlases : list of float arrays, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of the class.
        
    exponential_parameters: Parameters of the exponential likelihood distribution
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    #compute feature vectors for left and right lungs
    ### TODO: replace with likelihood params for class 0.


    length  = np.shape(probabilistic_atlases[0])[0]
    width = np.shape(probabilistic_atlases[0])[1]
    
    #Define pec area to segment
    pecPrior = probabilistic_atlases[0] + probabilistic_atlases[1] +\
       + probabilistic_atlases[2]+ probabilistic_atlases[3]
    zero_indeces_thresholding = pecPrior < 0.25 
    pecPriorSlicedialated = pecPrior
    pecPriorSlicedialated[zero_indeces_thresholding] = 0.0
    
    ones_indeces_thresholding = pecPrior > 0.24 
    pecPriorSlicedialated[ones_indeces_thresholding] = 1.0

    pecPriorSlicedialated = ndimage.binary_dilation(pecPriorSlicedialated, \
      iterations=8)
    ndimage.morphology.binary_fill_holes(pecPriorSlicedialated, \
      structure=None, output=pecPriorSlicedialated, origin=0)
    tozero_indeces_intensities = pecPriorSlicedialated < 1
           
    left_major_distance_map = compute_distance_to_atlas(probabilistic_atlases[0].astype(float))
    left_minor_distance_map = compute_distance_to_atlas(probabilistic_atlases[1].astype(float))     
    right_major_distance_map = compute_distance_to_atlas(probabilistic_atlases[2].astype(float))
    right_minor_distance_map = compute_distance_to_atlas(probabilistic_atlases[3].astype(float))
  
    #the exponential part of the likelihood, only distance
         
    left_major_polynomial_feature_map = PolynomialFeatureMap( [ \
      left_major_distance_map],[0] )  
    left_major_polynomial_feature_map.compute_num_terms()
    
    right_major_polynomial_feature_map = PolynomialFeatureMap( [ \
      right_major_distance_map],[0] )  
    right_major_polynomial_feature_map.compute_num_terms()


    left_minor_polynomial_feature_map = PolynomialFeatureMap( [ \
      left_minor_distance_map],[0] )  
    left_minor_polynomial_feature_map.compute_num_terms()
    
    right_minor_polynomial_feature_map = PolynomialFeatureMap( [ \
      right_minor_distance_map],[0] )  
    right_minor_polynomial_feature_map.compute_num_terms()
    
    #define the weights
    print("defining the weights, atlas shape is:")     
    print(np.shape(probabilistic_atlases))
    print(np.shape(input_image))
    
        # mu sigma lambda
    alpha_dleftmajor_given_leftmajor=expgaus_parameters[0]
    alpha_dleftminor_given_leftmajor=expgaus_parameters[1]
    alpha_drightmajor_given_leftmajor=expgaus_parameters[2]
    alpha_drightminor_given_leftmajor=expgaus_parameters[3]
    alpha_dleftmajor_given_leftminor=expgaus_parameters[4]
    alpha_dleftminor_given_leftminor=expgaus_parameters[5]
    alpha_drightmajor_given_leftminor=expgaus_parameters[6]
    alpha_drightminor_given_leftminor=expgaus_parameters[7]
    alpha_dleftmajor_given_rightmajor=expgaus_parameters[8]
    alpha_dleftminor_given_rightmajor=expgaus_parameters[9]
    alpha_drightmajor_given_rightmajor=expgaus_parameters[10]
    alpha_drightminor_given_rightmajor=expgaus_parameters[11]
    alpha_dleftmajor_given_rightminor=expgaus_parameters[12]
    alpha_dleftminor_given_rightminor=expgaus_parameters[13]
    alpha_drightmajor_given_rightminor=expgaus_parameters[14]
    alpha_drightminor_given_rightminor=expgaus_parameters[15]
    alpha_dleftmajor_given_non=expgaus_parameters[16]
    alpha_dleftminor_given_non=expgaus_parameters[17]
    alpha_drightmajor_given_non=expgaus_parameters[18]
    alpha_drightminor_given_non=expgaus_parameters[19]
    #left major
    
    
    ##something wrongwith this first one
    left_major_weights = [1.0]
    left_major_lambda = alpha_dleftmajor_given_leftmajor[2]
    left_major_weighted_density = ExpWeightedFeatureMapDensity([\
       left_major_distance_map], left_major_weights, \
       left_major_polynomial_feature_map, left_major_lambda)
    alpha_dleftmajor_given_leftmajor=[32.777230, 35.411380*35.411380, 0.327850]
    left_majorlikelihood =  stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftmajor_given_leftmajor[0], scale=alpha_dleftmajor_given_leftmajor[1]) 
       #left_major_weighted_density.compute()*\
    
    print("show likelihoos image")
    testimage = np.squeeze(input_image)
    im = plt.imshow(testimage, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()
        
    left_major_weights_given_left_minor  = [1.0 ]
    left_major_given_left_minor_lambda = alpha_dleftmajor_given_leftminor[2]
    left_major_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       left_major_distance_map], left_major_weights_given_left_minor, \
       left_major_polynomial_feature_map, left_major_given_left_minor_lambda)
    leftMajorIgivenLeftMinor = \
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftmajor_given_leftminor[0], scale=alpha_dleftmajor_given_leftminor[1])
        #left_major_given_leftminor_weighted_density.compute()*
        
    left_major_weights_given_right_major = [1.0 ]
    left_major_given_right_major_lambda = alpha_dleftmajor_given_rightmajor[2]
    leftmajor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
        left_major_distance_map], 
       left_major_weights_given_right_major, left_major_polynomial_feature_map, left_major_given_right_major_lambda)
    leftMajorIgivenRightMajor = leftmajor_given_rightmajor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftmajor_given_leftminor[0], scale=alpha_dleftmajor_given_leftminor[1])
    
    
    left_major_weights_given_right_minor = [1.0]
    left_major_given_right_minor_lambda = alpha_dleftmajor_given_rightminor[2]
    leftmajor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       left_major_distance_map], 
       left_major_weights_given_right_minor, left_major_polynomial_feature_map, left_major_given_right_minor_lambda)
    leftMajorIgivenRightMinor = leftmajor_given_rightminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftmajor_given_rightminor[0], scale=alpha_dleftmajor_given_rightminor[1])
    
    left_major_weights_given_non = [1.0 ]
    left_major_given_non_lambda = alpha_dleftmajor_given_non[2]
    leftmajor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       left_major_distance_map], 
       left_major_weights_given_non, left_major_polynomial_feature_map, left_major_given_non_lambda)
    leftMajorIgivenNon = leftmajor_given_non_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftmajor_given_non[0], scale=alpha_dleftmajor_given_non[1])
    
    #left minor
    left_minor_weights_given_left_major = [1.0 ]
    left_minor_lambda = alpha_dleftminor_given_leftmajor[2]
    left_minor_given_left_major_weighted_density = ExpWeightedFeatureMapDensity([\
       left_minor_distance_map], left_minor_weights_given_left_major, \
       left_minor_polynomial_feature_map, left_minor_lambda)
    leftMinorIgivenLeftMajor = left_minor_given_left_major_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftminor_given_leftmajor[0], scale=alpha_dleftminor_given_leftmajor[1])
    
    
    left_minor_weights = [1.0  ]
    left_minor_given_leftminor_lambda = alpha_dleftminor_given_leftminor[2]
    left_minor_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       left_minor_distance_map], left_minor_weights, \
       left_minor_polynomial_feature_map, left_minor_given_leftminor_lambda)
    left_minorlikelihood = left_minor_given_leftminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftminor_given_leftminor[0], scale=alpha_dleftminor_given_leftminor[1])
        

    left_minor_weights_given_right_major = [1.0  ]
    left_minor_given_rightmajor_lambda = alpha_dleftminor_given_rightmajor[2]
    leftminor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       left_minor_distance_map], 
       left_minor_weights_given_right_major, left_minor_polynomial_feature_map, left_minor_given_rightmajor_lambda)
    leftMinorIgivenRightMajor = leftminor_given_rightmajor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftminor_given_rightmajor[0], scale=alpha_dleftminor_given_rightmajor[1])
    
    left_minor_weights_given_right_minor = [1.0 ]
    left_minor_given_rightminor_lambda = alpha_dleftminor_given_rightminor[2]
    leftminor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       left_minor_distance_map], 
       left_minor_weights_given_right_minor, left_minor_polynomial_feature_map, left_minor_given_rightminor_lambda)
    leftMinorIgivenRightMinor = leftminor_given_rightminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftminor_given_rightminor[0], scale=alpha_dleftminor_given_rightminor[1])

    left_minor_weights_given_non = [1.0 ]
    left_minor_given_non_lambda = alpha_dleftminor_given_non[2]
    leftminor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       left_minor_distance_map], 
       left_minor_weights_given_non, left_minor_polynomial_feature_map, left_minor_given_non_lambda)
    leftMinorIgivenNon = leftminor_given_non_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_dleftminor_given_non[0], scale=alpha_dleftminor_given_non[1])          

    #right major
    
    right_major_weights_given_left_major = [1.0 ]
    right_major_given_left_major_lambda = alpha_drightmajor_given_leftmajor[2]
    rightmajor_given_leftmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       right_major_distance_map], right_major_weights_given_left_major, \
       right_major_polynomial_feature_map, right_major_given_left_major_lambda)
    rightMajorIgivenLeftMajor = rightmajor_given_leftmajor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightmajor_given_leftmajor[0], scale=alpha_drightmajor_given_leftmajor[1])
    
    right_major_weights_given_left_minor  = [1.0 ]
    right_major_given_left_minor_lambda = alpha_drightmajor_given_leftminor[2]
    right_major_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       right_major_distance_map], right_major_weights_given_left_minor, \
       left_major_polynomial_feature_map, right_major_given_left_minor_lambda)
    rightMajorIgivenLeftMinor = right_major_given_leftminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightmajor_given_leftminor[0], scale=alpha_drightmajor_given_leftminor[1])
    
          

    right_major_weights = [1.0  ]
    right_major_given_right_major_lambda = alpha_drightmajor_given_rightmajor[2]
    rightmajor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       right_major_distance_map], 
       right_major_weights, right_major_polynomial_feature_map, right_major_given_right_major_lambda)
    right_majorlikelihood = rightmajor_given_rightmajor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightmajor_given_rightmajor[0], scale=alpha_drightmajor_given_rightmajor[1])
    
    
    right_major_weights_given_right_minor = [1.0  ]
    right_major_given_right_minor_lambda = alpha_drightmajor_given_rightminor[2]
    rightmajor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       right_major_distance_map], 
       right_major_weights_given_right_minor, right_major_polynomial_feature_map, right_major_given_right_minor_lambda)
    rightMajorIgivenRightMinor = rightmajor_given_rightminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightmajor_given_rightminor[0], scale=alpha_drightmajor_given_rightminor[1])
    
    right_major_weights_given_non = [1.0 ]
    right_major_given_non_lambda = alpha_drightmajor_given_non[2]
    rightmajor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       right_major_distance_map], 
       right_major_weights_given_non, right_major_polynomial_feature_map, right_major_given_non_lambda)
    rightMajorIgivenNon = rightmajor_given_non_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightmajor_given_non[0], scale=alpha_drightmajor_given_non[1])
        
   #right minor
    right_minor_weights_given_left_major = [1.0  ]
    right_minor_given_left_major_lambda = alpha_drightminor_given_leftmajor[2]
    right_minor_given_left_major_weighted_density = ExpWeightedFeatureMapDensity([\
       right_minor_distance_map], right_minor_weights_given_left_major, \
       right_minor_polynomial_feature_map, right_minor_given_left_major_lambda)
    rightMinorIgivenLeftMajor = right_minor_given_left_major_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightminor_given_leftmajor[0], scale=alpha_drightminor_given_leftmajor[1])
    
 
    right_minor_weights_given_left_minor = [1.0 ]
    right_minor_given_left_minor_lambda = alpha_drightminor_given_leftminor[2]
    right_minor_given_leftminor_weighted_density = ExpWeightedFeatureMapDensity([\
       left_minor_distance_map], right_minor_weights_given_left_minor, \
       right_minor_polynomial_feature_map, right_minor_given_left_minor_lambda)
    rightMinorIgivenLeftMinor = right_minor_given_leftminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightminor_given_leftminor[0], scale=alpha_drightminor_given_leftminor[1])
        

    right_minor_weights_given_right_major = [1.0  ]
    right_minor_given_right_major_lambda = alpha_drightminor_given_rightmajor[2]
    rightminor_given_rightmajor_weighted_density = ExpWeightedFeatureMapDensity([\
       right_minor_distance_map], 
       right_minor_weights_given_right_major, right_minor_polynomial_feature_map, right_minor_given_right_major_lambda)
    rightMinorIgivenRightMajor = rightminor_given_rightmajor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightminor_given_rightmajor[0], scale=alpha_drightminor_given_rightmajor[1])
    
    right_minor_weights_given_right_minor = [1.0  ]
    right_minor_given_right_minor_lambda = alpha_drightminor_given_rightminor[2]
    rightminor_given_rightminor_weighted_density = ExpWeightedFeatureMapDensity([\
       right_minor_distance_map], 
       right_minor_weights_given_right_minor, right_minor_polynomial_feature_map, right_minor_given_right_minor_lambda)
    right_minorlikelihood = rightminor_given_rightminor_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightminor_given_rightminor[0], scale=alpha_drightminor_given_rightminor[1])
  
    right_minor_weights_given_non = [1.0 ]
    right_minor_given_non_lambda = alpha_drightminor_given_non[2]
    rightminor_given_non_weighted_density = ExpWeightedFeatureMapDensity([\
       right_minor_distance_map], 
       right_minor_weights_given_non, right_minor_polynomial_feature_map, right_minor_given_non_lambda)
    rightMinorIgivenNon = rightminor_given_non_weighted_density.compute()*\
       stats.norm.pdf(input_image.astype(np.float), loc=alpha_drightminor_given_non[0], scale=alpha_drightminor_given_non[1])
   
    print("computing probabilities")
    lungPrior = np.ones((length, width,1)).astype(np.float)
    lungPrior = np.add(probabilistic_atlases[0],probabilistic_atlases[1])
    lungPrior = np.add(lungPrior,probabilistic_atlases[2])
    lungPrior = np.add(lungPrior,probabilistic_atlases[3])
    
    notLungPrior = np.ones((length, width,1)).astype(np.float)
    notLungPrior = notLungPrior - lungPrior;
    
 
    p_I_dleftmajor = np.add(
         np.multiply(left_majorlikelihood.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),
         np.multiply( leftMajorIgivenLeftMinor.astype(np.float),\
         probabilistic_atlases[1].astype(np.float)))
    p_I_dleftmajor = np.add(p_I_dleftmajor,              
         np.multiply( leftMajorIgivenRightMajor.astype(np.float),
         probabilistic_atlases[2].astype(np.float)))
    p_I_dleftmajor = np.add(p_I_dleftmajor,                   
         np.multiply( leftMajorIgivenRightMinor.astype(np.float),\
         probabilistic_atlases[3].astype(np.float)))     
    p_I_dleftmajor = np.add(p_I_dleftmajor, 
         np.multiply(leftMajorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float))
         )  


    p_I_dleftminor = np.add(np.multiply(leftMinorIgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         left_minorlikelihood.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))
    p_I_dleftminor = np.add(p_I_dleftminor, np.multiply( \
         leftMinorIgivenRightMajor.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))      
    p_I_dleftminor = np.add(p_I_dleftminor,      np.multiply( \
         leftMinorIgivenRightMinor.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))
    p_I_dleftminor = np.add(p_I_dleftminor, np.multiply(leftMinorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))           
               
    p_I_drightmajor = np.add(np.multiply(rightMajorIgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         rightMajorIgivenLeftMinor.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))         
    p_I_drightmajor = np.add(p_I_drightmajor,     np.multiply( \
         right_majorlikelihood.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))       
    p_I_drightmajor = np.add(p_I_drightmajor,      np.multiply( \
         rightMajorIgivenRightMinor.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))        
    p_I_drightmajor = np.add(p_I_drightmajor,np.multiply(rightMajorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))                      
                                   
    p_I_drightminor = np.add(np.multiply(rightMinorIgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         rightMinorIgivenLeftMinor.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))
    p_I_drightminor = np.add(p_I_drightminor, np.multiply( \
         rightMinorIgivenRightMajor.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))  
    p_I_drightminor = np.add(p_I_drightminor, np.multiply( \
         right_minorlikelihood.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))    
    p_I_drightminor = np.add(p_I_drightminor, np.multiply(rightMinorIgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))             
    
    zero_indeces = (p_I_dleftmajor == 0)
    p_I_dleftmajor[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces = (p_I_dleftminor == 0)
    p_I_dleftminor[zero_indeces] = 0.000000000000000000000001
    
    zero_indeces = (p_I_drightmajor == 0)
    p_I_drightmajor[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces = (p_I_drightminor == 0)
    p_I_drightminor[zero_indeces] = 0.000000000000000000000001
    
    left_majorlikelihood[tozero_indeces_intensities]=0
    right_majorlikelihood[tozero_indeces_intensities]=0
    left_minorlikelihood[tozero_indeces_intensities]=0
    right_minorlikelihood[tozero_indeces_intensities]=0
   
   
    right_atlas_filename_major = "/Users/rolaharmouche/Documents/Data/COPDGene/10004O/10004O_INSP_STD_BWH_COPD/10004O_INSP_STD_BWH_COPD_leftPecMajorlikelihood.nrrd"
    nrrd.write(right_atlas_filename_major, left_majorlikelihood)       

    print("array shapes are:")
    print(np.shape(left_majorlikelihood))
    print(np.shape(p_I_dleftminor))
    print(np.shape(probabilistic_atlases))
    #segment given feature vectors
    segmented_labels = segment_chest_with_atlas([left_majorlikelihood.astype( \
       np.float), left_minorlikelihood.astype(np.float), right_majorlikelihood.astype(np.float), right_minorlikelihood.astype(np.float)], probabilistic_atlases, \
       [p_I_dleftmajor.astype(np.float), p_I_dleftminor.astype(np.float),p_I_drightmajor.astype(np.float), p_I_drightminor.astype(np.float)])
    
    return segmented_labels

def segment_pec_with_atlas_newfun(input_image, probabilistic_atlases, expgaus_parameters): 
    #TODO: and command line option to this command
    
    """Segment pecs using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    probabilistic_atlases : list of float arrays, shape (L, M, N)
        Atlas to use as segmentation prior. Each voxel should have a value in
        the interval [0, 1], indicating the probability of the class.
        
    exponential_parameters: Parameters of the exponential likelihood distribution
        ...
        
    Returns
    -------
    label_map : array, shape (L, M, N)
        Segmented image with labels adhering to CIP conventions
    """
    
    #compute feature vectors for left and right lungs
    ### TODO: replace with likelihood params for class 0.

    print("show input image")
    testimage = np.squeeze(input_image)
    im = plt.imshow(testimage, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()

    length  = np.shape(probabilistic_atlases[0])[0]
    width = np.shape(probabilistic_atlases[0])[1]
    
    #Define pec area to segment
    pecPrior = probabilistic_atlases[0] + probabilistic_atlases[1] +\
       + probabilistic_atlases[2]+ probabilistic_atlases[3]
    zero_indeces_thresholding = pecPrior < 0.25 
    pecPriorSlicedialated = pecPrior
    pecPriorSlicedialated[zero_indeces_thresholding] = 0.0
    
    ones_indeces_thresholding = pecPrior > 0.24 
    pecPriorSlicedialated[ones_indeces_thresholding] = 1.0

    pecPriorSlicedialated = ndimage.binary_dilation(pecPriorSlicedialated, \
      iterations=8)
    ndimage.morphology.binary_fill_holes(pecPriorSlicedialated, \
      structure=None, output=pecPriorSlicedialated, origin=0)
    tozero_indeces_intensities = pecPriorSlicedialated < 1
           
    left_major_distance_map = compute_distance_to_atlas(probabilistic_atlases[0].astype(float))
    left_minor_distance_map = compute_distance_to_atlas(probabilistic_atlases[1].astype(float))     
    right_major_distance_map = compute_distance_to_atlas(probabilistic_atlases[2].astype(float))
    right_minor_distance_map = compute_distance_to_atlas(probabilistic_atlases[3].astype(float))
  
    
    #define the weights
    print("defining the weights, atlas shape is:")     
    print(np.shape(probabilistic_atlases))
    print(np.shape(input_image))
    
        # mu sigma lambda
    alpha_dleftmajor_given_leftmajor=expgaus_parameters[0]
    alpha_dleftminor_given_leftmajor=expgaus_parameters[1]
    alpha_drightmajor_given_leftmajor=expgaus_parameters[2]
    alpha_drightminor_given_leftmajor=expgaus_parameters[3]
    alpha_dleftmajor_given_leftminor=expgaus_parameters[4]
    alpha_dleftminor_given_leftminor=expgaus_parameters[5]
    alpha_drightmajor_given_leftminor=expgaus_parameters[6]
    alpha_drightminor_given_leftminor=expgaus_parameters[7]
    alpha_dleftmajor_given_rightmajor=expgaus_parameters[8]
    alpha_dleftminor_given_rightmajor=expgaus_parameters[9]
    alpha_drightmajor_given_rightmajor=expgaus_parameters[10]
    alpha_drightminor_given_rightmajor=expgaus_parameters[11]
    alpha_dleftmajor_given_rightminor=expgaus_parameters[12]
    alpha_dleftminor_given_rightminor=expgaus_parameters[13]
    alpha_drightmajor_given_rightminor=expgaus_parameters[14]
    alpha_drightminor_given_rightminor=expgaus_parameters[15]
    alpha_dleftmajor_given_non=expgaus_parameters[16]
    alpha_dleftminor_given_non=expgaus_parameters[17]
    alpha_drightmajor_given_non=expgaus_parameters[18]
    alpha_drightminor_given_non=expgaus_parameters[19]
    #left major
    

    
    ##something wrongwith this first one
    left_majorlikelihood =  compute_new_likelihood(input_image, left_major_distance_map, alpha_dleftmajor_given_leftmajor) 
    leftMajordgivenLeftMinor = compute_new_likelihood(input_image, left_major_distance_map, alpha_dleftmajor_given_leftminor) #p(I,d_{leftmajor}|LeftMinor)
    leftMajordgivenRightMajor =  compute_new_likelihood(input_image, left_major_distance_map, alpha_dleftmajor_given_rightmajor)    
    leftMajordgivenRightMinor = compute_new_likelihood(input_image, left_major_distance_map, alpha_dleftmajor_given_rightminor)
    leftMajordgivenNon =  compute_new_likelihood(input_image, left_major_distance_map, alpha_dleftmajor_given_non)
     
    #left minor
    leftMinordgivenLeftMajor =  compute_new_likelihood(input_image, left_minor_distance_map, alpha_dleftminor_given_leftmajor) 
    left_minorlikelihood = compute_new_likelihood(input_image, left_minor_distance_map, alpha_dleftminor_given_leftminor)
    leftMinordgivenRightMajor =  compute_new_likelihood(input_image, left_minor_distance_map, alpha_dleftminor_given_rightmajor)    
    leftMinordgivenRightMinor = compute_new_likelihood(input_image, left_minor_distance_map, alpha_dleftminor_given_rightminor)
    leftMinordgivenNon =  compute_new_likelihood(input_image, left_minor_distance_map, alpha_dleftminor_given_non)
    
    #right major
    rightMajordgivenLeftMajor =  compute_new_likelihood(input_image, right_major_distance_map, alpha_drightmajor_given_leftmajor) 
    rightMajordgivenLeftMinor = compute_new_likelihood(input_image, right_major_distance_map, alpha_drightmajor_given_leftminor) #p(I,d_{leftmajor}|LeftMinor)
    right_majorlikelihood =  compute_new_likelihood(input_image, right_major_distance_map, alpha_drightmajor_given_rightmajor)    
    rightMajordgivenRightMinor = compute_new_likelihood(input_image, right_major_distance_map, alpha_drightmajor_given_rightminor)
    rightMajordgivenNon =  compute_new_likelihood(input_image, right_major_distance_map, alpha_drightmajor_given_non)
    
    #right minor
    rightMinordgivenLeftMajor =  compute_new_likelihood(input_image, right_minor_distance_map, alpha_drightminor_given_leftmajor) 
    rightMinordgivenLeftMinor = compute_new_likelihood(input_image, right_minor_distance_map, alpha_drightminor_given_leftminor)
    rightMinordgivenRightMajor =  compute_new_likelihood(input_image, right_minor_distance_map, alpha_drightminor_given_rightmajor)    
    right_minorlikelihood = compute_new_likelihood(input_image, right_minor_distance_map, alpha_drightminor_given_rightminor)
    rightMinordgivenNon =  compute_new_likelihood(input_image, right_minor_distance_map, alpha_drightminor_given_non)
    
    print("show likelihood image")
    testimage = np.squeeze(left_majorlikelihood)
    im = plt.imshow(testimage, cmap='hot')
    plt.colorbar(im, orientation='horizontal')
    plt.show()
    
    print("computing probabilities")
    lungPrior = np.ones((length, width,1)).astype(np.float)
    lungPrior = np.add(probabilistic_atlases[0],probabilistic_atlases[1])
    lungPrior = np.add(lungPrior,probabilistic_atlases[2])
    lungPrior = np.add(lungPrior,probabilistic_atlases[3])
    
    notLungPrior = np.ones((length, width,1)).astype(np.float)
    notLungPrior = notLungPrior - lungPrior;
    
 
    p_I_dleftmajor = np.add(
         np.multiply(left_majorlikelihood.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),
         np.multiply( leftMajordgivenLeftMinor.astype(np.float),\
         probabilistic_atlases[1].astype(np.float)))
    p_I_dleftmajor = np.add(p_I_dleftmajor,              
         np.multiply( leftMajordgivenRightMajor.astype(np.float),
         probabilistic_atlases[2].astype(np.float)))
    p_I_dleftmajor = np.add(p_I_dleftmajor,                   
         np.multiply( leftMajordgivenRightMinor.astype(np.float),\
         probabilistic_atlases[3].astype(np.float)))     
    p_I_dleftmajor = np.add(p_I_dleftmajor, 
         np.multiply(leftMajordgivenNon.astype(np.float), \
         notLungPrior.astype(np.float))
         )  


    p_I_dleftminor = np.add(np.multiply(leftMinordgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         left_minorlikelihood.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))
    p_I_dleftminor = np.add(p_I_dleftminor, np.multiply( \
         leftMinordgivenRightMajor.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))      
    p_I_dleftminor = np.add(p_I_dleftminor,      np.multiply( \
         leftMinordgivenRightMinor.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))
    p_I_dleftminor = np.add(p_I_dleftminor, np.multiply(leftMinordgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))           
               
    p_I_drightmajor = np.add(np.multiply(rightMajordgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         rightMajordgivenLeftMinor.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))         
    p_I_drightmajor = np.add(p_I_drightmajor,     np.multiply( \
         right_majorlikelihood.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))       
    p_I_drightmajor = np.add(p_I_drightmajor,      np.multiply( \
         rightMajordgivenRightMinor.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))        
    p_I_drightmajor = np.add(p_I_drightmajor,np.multiply(rightMajordgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))                      
                                   
    p_I_drightminor = np.add(np.multiply(rightMinordgivenLeftMajor.astype(np.float), \
         probabilistic_atlases[0].astype(np.float)),np.multiply( \
         rightMinordgivenLeftMinor.astype(np.float),probabilistic_atlases[1].astype( \
         np.float)))
    p_I_drightminor = np.add(p_I_drightminor, np.multiply( \
         rightMinordgivenRightMajor.astype(np.float),probabilistic_atlases[2].astype( \
         np.float)))  
    p_I_drightminor = np.add(p_I_drightminor, np.multiply( \
         right_minorlikelihood.astype(np.float),probabilistic_atlases[3].astype( \
         np.float)))    
    p_I_drightminor = np.add(p_I_drightminor, np.multiply(rightMinordgivenNon.astype(np.float), \
         notLungPrior.astype(np.float)))             
    
    print("array shapes are:")
    print(np.shape(left_majorlikelihood))
    print(np.shape(p_I_dleftminor))
    print(np.shape(probabilistic_atlases))
    
    zero_indeces = (p_I_dleftmajor == 0)
    p_I_dleftmajor[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces = (p_I_dleftminor == 0)
    p_I_dleftminor[zero_indeces] = 0.000000000000000000000001
    
    zero_indeces = (p_I_drightmajor == 0)
    p_I_drightmajor[zero_indeces] = 0.000000000000000000000001
   
    zero_indeces = (p_I_drightminor == 0)
    p_I_drightminor[zero_indeces] = 0.000000000000000000000001
    
    left_majorlikelihood[tozero_indeces_intensities]=0
    right_majorlikelihood[tozero_indeces_intensities]=0
    left_minorlikelihood[tozero_indeces_intensities]=0
    right_minorlikelihood[tozero_indeces_intensities]=0
   
   
    right_atlas_filename_major = "/Users/rolaharmouche/Documents/Data/COPDGene/10004O/10004O_INSP_STD_BWH_COPD/10004O_INSP_STD_BWH_COPD_leftPecMajorlikelihood.nrrd"
    nrrd.write(right_atlas_filename_major, left_majorlikelihood)#left_majorlikelihood)       


    #segment given feature vectors
    segmented_labels = segment_chest_with_atlas([left_majorlikelihood.astype( \
       np.float), left_minorlikelihood.astype(np.float), right_majorlikelihood.astype(np.float), right_minorlikelihood.astype(np.float)], probabilistic_atlases, \
       [p_I_dleftmajor.astype(np.float), p_I_dleftminor.astype(np.float),p_I_drightmajor.astype(np.float), p_I_drightminor.astype(np.float)])
    
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
    if (np.size(np.shape(structure_posterior_energy)[1]) == 3):
        num_slices = np.shape(structure_posterior_energy)[2];
    else:
        num_slices = 1 
        
    numNodes = length * width
    segmented_image = np.zeros((length, width, num_slices), dtype = np.int32)
    
    for slice_num in range(0, num_slices):
        print("graph cut slice" +str(slice_num))
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

def compute_distance_to_atlas(atlas):
    """Computes the Eucledian distance to a thresholded probabilistic atlas   
     
    Parameters
    ----------
    atlas: A float array with shape (L, M, N) 
            representing the probabilistic atlas 
        ...
        
    Returns
    -------
    atlas_distance_map : A float array with shape (L, M, N) 
        Contains distances to the thresholded atlas
    """
    
    zero_indeces_ll = atlas < 0.5
    leftLungPriorthres = np.ones((np.shape(atlas)), dtype=float) 
    leftLungPriorthres[zero_indeces_ll] = 1.0    
    one_indeces_ll = atlas >= 0.5
    leftLungPriorthres[one_indeces_ll] = 0.0      
    atlas_distance_map = \
        ndimage.morphology.distance_transform_edt(leftLungPriorthres)
            
    zero_indeces = (atlas_distance_map == 0)
    atlas_distance_map[zero_indeces] = 0.000000000000000000000001
    
    return atlas_distance_map

def compute_new_likelihood(intensity_data, distance_data, x):
    
    imshape = np.shape(intensity_data)
    
    
    pow1 = pow((intensity_data[:] - (x[0]*distance_data[:]+x[1])),2)
    
    numerator = np.exp(-(pow((np.ravel(intensity_data) - (x[0]*np.ravel(distance_data)+x[1])),2)/(2*pow((x[2]*np.ravel(distance_data)+x[3]),2)) + x[4]*np.ravel(distance_data)))
    denom = (x[2]+x[3]*x[4])*np.sqrt(2*np.pi)/(x[4])*(x[4])
    myfun2 = numerator/denom
       
    myfun3 = np.reshape(myfun2, (imshape[0],imshape[1],imshape[2]))
    return myfun3