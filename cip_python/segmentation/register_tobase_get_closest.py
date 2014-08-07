import os
import nrrd
import subprocess
import numpy as np
from scipy import ndimage
from cip_python.utils.get_closest_cases import getClosestCases

def clean_ct(CT_original):
    CT_clean= CT_original.split('.')[0]+"_cleaned."+\
        CT_original.split('.')[1] 
           
    ct_ori, imageInfo= nrrd.read(CT_original)          
    ct_cl, nr_objects = ndimage.label(ct_ori > -700) 
    ct_ori[ct_cl != 1]=-1024
    nrrd.write(CT_clean, np.squeeze(ct_ori))
        
def register_2d_ct(moving_CT_original, fixed_CT_original, output_transfo_name):

    """
    Function that registers a moving 2D ct image to a fixed image, and saves
    the resulting transformation in a .tfm file
    
    moving_CT_original : inout moving CT image
    
    fixed_CT_original : input fixed ct image

    output_transfo_name : filename of the output transformation to be saved.
    
    (Mask files will be saved in the same directory as the input files)
    
    """        
    
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print path_name + " environment variable is not set"
            exit()
            
    """
    Remove cruft from images before performing registration
    """
    moving_CT= moving_CT_original.split('.')[0]+"_cleaned."+\
        moving_CT_original.split('.')[1]

    fixed_CT= fixed_CT_original.split('.')[0]+"_cleaned."+\
        fixed_CT_original.split('.')[1]
            
    ct_ori, imageInfo= nrrd.read(moving_CT_original)          
    ct_cl, nr_objects = ndimage.label(ct_ori > -700) 
    ct_ori[ct_cl != 1]=-1024
    nrrd.write(moving_CT, np.squeeze(ct_ori))
    
    moving_mask_rigid= moving_CT_original.split('.')[0]+"_registrationMask."+\
        moving_CT_original.split('.')[1]

    rigid_mask = np.zeros_like(ct_ori)         
    rigid_mask[ct_ori<(-1023)] = 0
    rigid_mask[ct_ori>(-1024)] = 1
    nrrd.write(moving_mask_rigid, np.squeeze(rigid_mask))
    
    ct_ori, imageInfo= nrrd.read(fixed_CT_original)          
    ct_cl, nr_objects = ndimage.label(ct_ori > -700) 
    ct_ori[ct_cl != 1]=-1024
    nrrd.write(fixed_CT, np.squeeze(ct_ori))

    registerCall = os.path.join(path['CIP_PATH'],"RegisterCT2D")
    register_call = registerCall+" -m "+moving_CT+" -f "+\
                fixed_CT+ " --outputTransform "+output_transfo_name+\
                 " --isIntensity --movingLabelmapFileName "+moving_mask_rigid
            
    print(register_call)  
    os.system(register_call)

def compute_ct_mask_similarity_withlabel(input_ct_volume, input_labelmap_filename,\
    output_ct_filename, output_maskfilename = None):

        cropped_data_temp, imageInfo= nrrd.read(input_labelmap_filename)
        above_zero = cropped_data_temp>0
        belowthresh = cropped_data_temp<17000 #fat is 17944
    
        #threshold slice to contain only all pec data
        cropped_data_temp[above_zero & belowthresh ] = 1
        cropped_data_temp[cropped_data_temp>1] = 0   
    
        #dilate
    	cropped_data_temp = np.array(ndimage.binary_dilation(cropped_data_temp, iterations = 20)).astype(np.int) #10    
        cropped_data = np.squeeze(cropped_data_temp)        
    
        ##now find boundingbox
        b = np.where(cropped_data>0)
        cropped_data[:] = -1024
        cropped_data[min(b[0]):max(b[0])+1, min(b[1]):max(b[1])+1 ] = \
            input_ct_volume[min(b[0]):max(b[0])+1, min(b[1]):max(b[1])+1 ] 

        #label all -1024 as 0, because resampling messes that up
        if (output_maskfilename != None):
            sim_mask = np.zeros_like(cropped_data)
            sim_mask[cropped_data>(-1024)] = 1
            nrrd.write(output_maskfilename,np.squeeze(sim_mask))
        
        cropped_data[cropped_data<(-1023)] = 0
        nrrd.write(output_ct_filename,np.squeeze(cropped_data))
        
def compute_ct_mask_similarity(input_labelmap_filename, output_maskfilename):

        cropped_data_temp, imageInfo= nrrd.read(input_labelmap_filename)
        above_zero = cropped_data_temp>0
        belowthresh = cropped_data_temp<17000 #fat is 17944
    
        #threshold slice to contain only all pec data
        cropped_data_temp[above_zero & belowthresh ] = 1
        cropped_data_temp[cropped_data_temp>1] = 0   
    
        #dilate
    	cropped_data_temp = np.array(ndimage.binary_dilation(cropped_data_temp,\
    	   iterations = 20)).astype(np.int) #10    
        cropped_data = np.squeeze(cropped_data_temp)        
    
        ##now find boundingbox
        b = np.where(cropped_data>0)
        cropped_data[min(b[0]):max(b[0])+1, min(b[1]):max(b[1])+1 ] = 1
        nrrd.write(output_maskfilename,np.squeeze(cropped_data))
        
        
def compute_similarity_from_filenames(testing_ct_filename, \
    training_case_ct_filenames,training_case_label_filenames, list_of_transfos, \
    base_case_transfo = None):
    
    """
    generalized call to get compute similarity so that we always call the 
    same function. if  base_case_transfo = Null, then 1 transformation.
    
    make sure identity base case transfos exist.  
    
    The assumption is that base to training always exists. So we don't need 
    to invert transformations        
    """  
    
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print path_name + " environment variable is not set"
            exit()
    
    GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
        "GetTransformationSimilarityMetric2D")
        
    list_of_similarity_files = [""]*len(training_case_ct_filenames)
    
    for x in range(0,len(training_case_ct_filenames)):  
        
        # compute mask and cropped image, save in temp_dir
        moving_mask_filename= training_case_ct_filenames[x].split('.')[0]+\
            "_similarityMask."+training_case_ct_filenames[x].split('.')[1]   
            
        #leave those as is for now. I think it did make a diff with final
        #result though.      
        moving_cropped_ct_filename = training_case_ct_filenames[x].split('.')[0]+\
            "_ctCroppedMask."+training_case_ct_filenames[x].split('.')[1]
        fixed_cropped_ct_filename =testing_ct_filename.split('.')[0]+\
            "_ctCroppedMask."+testing_ct_filename.split('.')[1]

        clean_ct(training_case_ct_filenames[x])
        clean_ct(testing_ct_filename)
        moving_CT= training_case_ct_filenames[x].split('.')[0]+"_cleaned."+\
            training_case_ct_filenames[x].split('.')[1]

        fixed_CT= testing_ct_filename.split('.')[0]+"_cleaned."+\
            testing_ct_filename.split('.')[1]
        
        compute_ct_mask_similarity(training_case_label_filenames[x],\
            moving_mask_filename)
                                    
        if (base_case_transfo == None):
            list_of_similarity_files[x] = list_of_transfos[x]+ "_measures.xml"
            transfos_for_similarity = list_of_transfos[x]
        else:
            list_of_similarity_files[x] = base_case_transfo+"_followedby_"+\
               list_of_transfos[x].split('/')[-1]+"_measures.xml"
            transfos_for_similarity = list_of_transfos[x]+","+base_case_transfo
                                            
        similarity_call= GetTransformationSimilarityMetric+ \
            " --fixedCTFileName "+fixed_CT +\
            " --movingCTFileName "+ moving_CT+ \
            " --movingLabelMapFileName "+ moving_mask_filename+\
            " --inputTransform "+transfos_for_similarity+\
            " --outputXMLFile "+list_of_similarity_files[x]+\
            " --SimilarityMetric nc"    
                       
        print(similarity_call)                
        subprocess.call(similarity_call, shell=True);
        
    return list_of_similarity_files
                                                     
def register_tobase_get_closest( testing_ct_filename, base_case_ct_filenames, 
    base_case_label_filenames, test_case_transfo_dir, is_register):

    """
    registers the test ct scan to each of the base_case_ct_filenames
    and finds the closest cagse according to some similarity metric
    """
    
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print path_name + " environment variable is not set"
            exit()
    
    base_to_testing_transfo_names = [""]*len(base_case_ct_filenames)                
    print(test_case_transfo_dir)
    for ii in range(0, len(base_case_ct_filenames)) :     
        base_case_ct_filename =  base_case_ct_filenames[ii]  
        base_to_testing_transfo_names[ii] = os.path.join(test_case_transfo_dir, \
        base_case_ct_filenames[ii].split('/')[-1].split('.')[0]+"_to_"+testing_ct_filename.split('/')[-1].split('.')[0]+"_tfm_0GenericAffine.tfm")             
        print(base_to_testing_transfo_names[ii])
        if(is_register is True):
            register_2d_ct( base_case_ct_filename, testing_ct_filename,  base_to_testing_transfo_names[ii])
                
   
    """
    find closest base case label filename and the index in the list we have
    """ 
    training_similarity_files = compute_similarity_from_filenames(testing_ct_filename, \
        base_case_ct_filenames, base_case_label_filenames, \
        base_to_testing_transfo_names)
    
    closest_case_ct = getClosestCases(base_case_ct_filenames, \
            training_similarity_files, "ncc", 1, 1)   
        
    
    """
    Return the ct filename of the closest base case and the corresponding 
    closest transformation
    """
    return closest_case_ct[0,0], base_to_testing_transfo_names[\
        base_case_ct_filenames.index(closest_case_ct[0,0])]                   
 

    