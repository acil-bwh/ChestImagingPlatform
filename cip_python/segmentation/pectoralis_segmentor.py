import pickle

import nrrd
import numpy as np

from cip_python.segmentation.segment_chest_with_atlas import segment_pec_with_atlas

class pectoralis_segmentor:
    def __init__(self, input_volume,  testing_ct_filename, training_labelmaps_filenames,\
            training_ct_filenames, base_case_ct_filenames, base_case_labelmap_filenames, \
            test_case_transfo_dir, transformation_filenames,  num_closest_cases,\
            similarity, threshold_value_for_similarity):
        self._input_volume = input_volume
        #self._output_volume = output_volume
        self._testing_ct_filename = testing_ct_filename
        self._training_labelmaps_filenames = training_labelmaps_filenames
        self._training_ct_filenames = training_ct_filenames
        self._base_case_ct_filenames = base_case_ct_filenames
        self._base_case_labelmap_filenames = base_case_labelmap_filenames
        self._test_case_transfo_dir = test_case_transfo_dir
        self._transformation_filenames = transformation_filenames
        self._num_closest_cases = num_closest_cases
        self._similarity = similarity
        self._threshold_value_for_similarity = threshold_value_for_similarity
        self._AllClasses = ["leftmajor","leftminor","rightmajor","rightminor","nonpec"]
        self._PecClasses = ["leftmajor","leftminor","rightmajor","rightminor"]
        self._pec_label_postfix = "_pecsSubqFatClosedSlice"
        self._transform_postfix = "_pecsSubqFat"
        self._mask_postfix = "_pecSlice_thresholded"
    
    def rev(a, axis = -1):
        a = np.asarray(a).swapaxes(axis, 0)
        a = a[::-1,...]
        a = a.swapaxes(0, axis)
        return a
        
    def execute(self):
        
        image_dimensions = np.shape(self._input_volume)
                                                                                                                                                                                                                                                                                                                           
        prior_probabilities = dict()
        for class_name in self._AllClasses: #define non pec prior and compute later  
            prior_probabilities[class_name] = np.zeros((image_dimensions[0], \
                image_dimensions[1],image_dimensions[2]), dtype=np.float)   
        #prior_probabilities = construct_pec_atlas_from_filenames.compute_atlas_from_labelfiles(\
        #    self._input_volume, self._testing_ct_filename, self._training_ct_filenames,\
        #    self._training_labelmaps_filenames, self._base_case_ct_filenames, \
        #    self._base_case_labelmap_filenames,\
        #    self._test_case_transfo_dir, self._transformation_filenames, \
        #    self._num_closest_cases, self._similarity, \
        #    self._threshold_value_for_similarity)
    
        ### for debugging purposes, read atlas:
        for class_index in self._PecClasses:
            atlas_filename_temp = "/Users/rolaharmouche/Documents/Data/COPDGene/10010J/10010J_INSP_STD_NJC_COPD/10010J_INSP_STD_NJC_COPD_"+class_index+"_atlas_nc_multiplebase15closest_threshold0_6.nrrd"    
            prior_probabilities[class_index], test= nrrd.read(atlas_filename_temp)
 
    
        print("atlases read..")
        #i=0
        #for class_name in self._PecClasses:
        #    test_read = vtk.vtkImageReader()
        #    test_read.SetFileName(training_labelmaps_content[i].rstrip("\n"))
        #    test_read.SetNumberOfScalarComponents(1)
        #    test_read.SetFileDimensionality(3)
        #    #test_read.SetDataExtent(0,image_dimensions[2]-1,0,image_dimensions[1]-1,0,image_dimensions[0]-1)
        #    test_read.SetDataExtent(0,image_dimensions[0]-1,0,image_dimensions[1]-1,0,image_dimensions[2]-1)
        #    data = test_read.GetOutput()
        #    data.Update()
        #    #print(data)
        #    prior_probabilities[class_name]=VN.vtk_to_numpy(\
        #        data.GetPointData().GetScalars()).reshape(image_dimensions[0],image_dimensions[1],image_dimensions[2])
        #        
        #    #prior_probabilities[class_name]=rev(a, axis = 2)    
        #    prior_shape = np.shape(prior_probabilities[class_name])
        #    i=i+1
        
        #load parameter files
        pkl_file = open("/Users/rolaharmouche/Documents/Data/distributions/postmiccai_complete_likelihood_params_v4.pkl", 'rb')        
        distribution_params = pickle.load(pkl_file)
        pkl_file.close()
        
        
        
        pec_segmentation_labels, out_posteriors = segment_pec_with_atlas(self._input_volume, prior_probabilities, distribution_params, \
            distribution_params, distribution_params, distribution_params, self._PecClasses, self._AllClasses)
        output_volume = np.zeros_like(pec_segmentation_labels["leftmajor"])   
        print(np.shape(output_volume))
        #do proper labling 
        left_major_indeces = (pec_segmentation_labels["leftmajor"] == 1)#13592
        output_volume[left_major_indeces] = 13592
        left_minor_indeces = (pec_segmentation_labels["leftminor"] == 1)#13336
        output_volume[left_minor_indeces] = 13336
        right_minor_indeces = (pec_segmentation_labels["rightminor"] == 1)#13335
        output_volume[right_minor_indeces] = 13335

        right_major_indeces = (pec_segmentation_labels["rightmajor"] == 1)#13591
        output_volume[right_major_indeces] = 13591

        #output_volume = prior_probabilities["leftminor"] #pec_segmentation_labels[0]
        #output_volume[0,:,:] = pec_segmentation_labels[0].squeeze()
        #for i in range(0,image_dimensions[0]):
        #    for j in range(0,image_dimensions[1]):
        #        #for k in range(0,image_dimensions[2]):
        #            #print(np.shape(pec_segmentation_labels))
        #            #print(pec_segmentation_labels[0][0][i][j])
        #            output_volume[i][j] = pec_segmentation_labels[0][0][i][j]
                    
        print("out volume shape in pectoralis segmentor, new")
        print(np.shape(output_volume))
        return output_volume, prior_probabilities, out_posteriors








