import os, sys, subprocess
import numpy as np
import nrrd
from cip_python.segmentation import register_tobase_get_closest
from cip_python.segmentation import compute_dice_similarity_from_filenames
from cip_python.segmentation import construct_probabilistic_atlas
# from .utils import getClosestCases

def compute_atlas_from_labelfiles(input_vol, testing_ct_filename, training_ct_filenames,\
    training_labelmap_filenames, base_case_ct_filenames, base_case_labelmap_filenames,\
    test_case_transfo_dir, transformation_filenames, num_closest_cases, \
    similarity,  threshold_value_for_similarity):

    """
    Compute atlas from a list of label files
    
    Inputs
    ------
    
    testing_ct_filename : full path of the ct filename to be segmented
    
    training_ct_filenames : full path of the ct filenames in the training data
        
    training_labelmap_filenames : full path of the labelmaps in the training data
        
    base_case_ct_filenames : filenames of the base case IDs
    
    test_case_transfo_dir : directory where the base to test transfos
        and the transformed labelmaps will be stored
        
    transformation_filenames["case ids"] : 1 line per labelmap filename. dict()
        transfo from base id to test case
	    
    num_closest_cases : number of cases to be selected for atlas generation

    similarity : string. type of similarity used to find closest cases
        
    threshold_value_for_similarity : value below which cases won't be chosen
    
    assumptions about naming and directories : 
        transformations to be computed in 1 specific folder : test_case_transfo_dir
        transfo filenames will be name1_to_name2_postfix.tfm
        resampled labelmaps : test_case_transfo_dir
        masks of training data : place in same directory as traning data    
    """

    # set environment variables and function calls ..
    toolsPaths = ['TEEM_PATH','ITKTOOLS_PATH','CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print path_name + " environment variable is not set"
            sys.exit()
            
    prior_atlas = dict()
    pec_classes = ["leftmajor","leftminor","rightmajor","rightminor"]

    label_value = dict()
    closest_label_maps = dict()
    closest_similarity_values = [0.0]*num_closest_cases
    
    label_value["leftmajor"] = 13592
    label_value["leftminor"] = 13336
    label_value["rightmajor"] = 13591
    label_value["rightminor"] = 13335
    label_value["nonpec"] = 0
    label_value["subqleft"] = 17944
    label_value["subqright"] = 17943
    
    resamplecall = os.path.join(path['CIP_PATH'], "ResampleLabelMap2D")
    
    """
    compute transformations to all base cases and find closest base case id
    Files required : full path ct file names of all base cases
    """
    closest_base_ct, close_base_tfm = register_tobase_get_closest(\
        testing_ct_filename, base_case_ct_filenames, base_case_labelmap_filenames, \
        test_case_transfo_dir, True)
        
    print("the closest base case is : "+closest_base_ct+ " " + close_base_tfm)
    
    base_case_key = closest_base_ct.split("/")[-1].split(".")[0].rstrip("\n")
    
    list_transfos_given_base = [""]*np.shape(transformation_filenames[base_case_key])[0]
    for i in range(0, np.shape(transformation_filenames[base_case_key])[0]):
        list_transfos_given_base[i] = transformation_filenames[base_case_key][i]
    
    for class_index in pec_classes: 
        
        print("constucting atlas for "+ class_index)
        """
        For each class, find the most similar cases by computing similarity.
        Files required : full path ct slices of all training data, full path masks of all training data (to be 
        generated here), transformations from base to all training data.
        """
        training_similarity_files = compute_dice_similarity_from_filenames(testing_ct_filename, \
            training_ct_filenames, training_labelmap_filenames, \
            list_transfos_given_base, close_base_tfm)
    
        closest_cases  = getClosestCases(training_labelmap_filenames, \
            training_similarity_files, "dice", num_closest_cases, threshold_value_for_similarity)  
        num_closest_cases_nonzero = np.shape(filter(None, closest_cases[0]))[0]

        closest_cases_tfms = [""]*num_closest_cases_nonzero
        for i in range(0,num_closest_cases_nonzero):
            closest_cases_tfms[i] = list_transfos_given_base[\
                training_labelmap_filenames.index(closest_cases[0,i])]

        imageFull, imageInfo= nrrd.read(testing_ct_filename)
    
        closest_label_maps[class_index] =[np.zeros([np.shape(imageFull)[0], \
            np.shape(imageFull)[1], np.shape(imageFull)[2]], dtype=float)] \
            *num_closest_cases_nonzero              

        """
        resample all closest cases
        Files required : labelmaps of all training data
        """
        for i in range(0,num_closest_cases_nonzero): 
            # we assume that this transformation exists
            training_to_base_tfm = closest_cases_tfms[i] 
            base_to_testing = close_base_tfm                                     
                                
       	
           	    
       	    source_file = closest_cases[0, i]
    	    
            template_file  = testing_ct_filename 
                
            dest_file = os.path.join(test_case_transfo_dir, \
                source_file.split("/")[-1]+"_to_"+template_file.split("/")[-1]+".nrrd")
                
            """
            multiple transformations required when the training case is not the base case. 
            """    

            if(close_base_tfm != closest_cases_tfms[i]):
    
                resamp_call = resamplecall+" -d "+template_file+" -r "+ \
                    dest_file+" -t "+training_to_base_tfm+","+ \
    	           base_to_testing+" -l "+source_file

            else: 
                """
                1 transformation required
                """            
                resamp_call =  resamplecall+" -d "+template_file+" -r "+ \
                    dest_file+" -t "+close_base_tfm +" -l "+source_file      
                               
            print(resamp_call) 
            subprocess.call(resamp_call, shell=True) 

            """ load the registered labelmap .
                exract left and right major/minor labelmaps and binarise. """   
           
            imageFulldest, imageInfo= nrrd.read(dest_file)
            
        
            closest_label_maps[class_index][i] =np.zeros([np.shape(imageFull)[0], \
                np.shape(imageFull)[1], np.shape(imageFull)[2]], dtype=float) 
                
            the_labels = imageFulldest!=label_value[class_index]
            closest_label_maps[class_index][i][the_labels] = 0.0 
            the_labels = imageFulldest==label_value[class_index]
            closest_label_maps[class_index][i][the_labels] = 1.0
                                 
            closest_similarity_values[i] = closest_cases[1, i]        
                                                
    """
    average
    """


    #normalise weights so that they sum to 1
    closest_similarity_values_float = np.array(map(float, \
        closest_similarity_values))/np.sum(np.array(map(float, \
        closest_similarity_values)))
                
    for class_index in pec_classes:
         prior_atlas[class_index] = construct_probabilistic_atlas(closest_label_maps[class_index], \
            True, np.array(map(float, closest_similarity_values_float)), None)

    """
    return atlases
    """
    #return prior_atlas["leftmajor"], prior_atlas["leftminor"], prior_atlas["rightmajor"],prior_atlas["rightminor"]
    return prior_atlas