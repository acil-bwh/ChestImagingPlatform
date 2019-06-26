import os
import nrrd
import subprocess
import numpy as np
from scipy import ndimage
from ..utils import getClosestCases

def clean_ct(CT_original, CT_clean):

           
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
            print (path_name + " environment variable is not set")
            exit()
            
    """
    Remove cruft from images before performing registration
    """
    moving_CT= moving_CT_original.split('.')[0]+"_cleaned."+\
        moving_CT_original.split('.')[1]

    fixed_CT= fixed_CT_original.split('.')[0]+"_cleaned."+\
        fixed_CT_original.split('.')[1]
            

    input_moving_mask_rigid= '_'.join(moving_CT_original.split("_")[0:-1])+"_pecsSubqFatClosedSlice."+\
        moving_CT_original.split('.')[1]
           
    moving_mask_rigid= '_'.join(moving_CT_original.split("_")[0:-1])+"_pecsSubqFatClosedSlice_thresholded."+\
        moving_CT_original.split('.')[1]

    mask, imageInfo= nrrd.read(input_moving_mask_rigid)          
    above_zero = mask>0
    belowthresh = mask<17000 #fat is 17944
    
        #threshold slice to contain only all pec data
    mask[above_zero & belowthresh ] = 1
    mask[mask>1] = 0  
        
    nrrd.write(moving_mask_rigid, np.squeeze(mask))
    
    clean_ct(moving_CT_original, moving_CT)
    clean_ct(fixed_CT_original, fixed_CT)
    
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
        
def compute_ct_mask_similarity(input_labelmap_filename, input_ctfilename , 
    output_maskfilename, dilation_value):

        cropped_data_temp, imageInfo= nrrd.read(input_labelmap_filename)
        above_zero = cropped_data_temp>0
        belowthresh = cropped_data_temp<17000 #fat is 17944
    
        #threshold slice to contain only all pec data
        cropped_data_temp[above_zero & belowthresh ] = 1
        cropped_data_temp[cropped_data_temp>1] = 0  
        
        
    
        #dilate
    	#cropped_data_temp = np.array(ndimage.binary_dilation(cropped_data_temp,\
    	#   iterations = 10)).astype(np.int) #10    
     #   cropped_data = np.squeeze(cropped_data_temp)  
        if (dilation_value > 0):        
    	    cropped_data_temp = np.array(ndimage.binary_dilation(cropped_data_temp,\
    	       iterations = dilation_value)).astype(np.int) #10 is the last functional value   
        cropped_data = np.squeeze(cropped_data_temp)        
        print(np.shape(cropped_data)) 
        #now find boundingbox
        #b = np.where(cropped_data>0)
        #cropped_data[min(b[0]):max(b[0])+1, min(b[1]):max(b[1])+1 ] = 1
        
        #remove lung tissue
        ct_data_temp, info = nrrd.read (input_ctfilename)
        ct_data = np.squeeze(ct_data_temp)
        print(np.shape(ct_data)) 
        lung_indeces = np.where(ct_data < (-1022))
        cropped_data[lung_indeces] = 0
        
        nrrd.write(output_maskfilename,np.squeeze(cropped_data))
        

def compute_edge_mask(input_mask_name, output_mask_name):
    """
    generates a mask that only takes into account the edge of the existing mask 
    with some dilation
    """
    toolsPaths = ['CIP_PATH', 'TEEM_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()    


    mask_data, options = nrrd.read(input_mask_name)

    #make the number of iterations proportional to the size of the pecs
    
    dilated_mask = np.array(ndimage.binary_dilation(mask_data, iterations = 10)).astype(np.int) 
    erdoded_mask = np.array(ndimage.binary_erosion(mask_data, iterations = 5)).astype(np.int) 
    
    final_array = np.zeros_like(mask_data)
    final_array = np.bitwise_and(dilated_mask, -erdoded_mask+1)
    nrrd.write(output_mask_name,final_array)
                
def compute_similarity_from_filenames(testing_ct_filename, \
    training_case_ct_filenames,training_case_label_filenames, list_of_transfos, \
    base_case_transfo = None):
    
    """
    generalized call to get compute similarity so that we always call the 
    same function. if  base_case_transfo = Null, then 1 transformation.
    
    make sure identity base case transfos exist.  
    
    The assumption is that base to training always exists. So we don't need 
    to invert transformations        
    
    the label filenames are supposed to be already thresholded to the right pec
    """  
    
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()
    
    print("\n\n\n In compute_similarity_from_filenames \n\n\n")
    GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
        "GetTransformationSimilarityMetric2D")
        
    list_of_similarity_files = [""]*len(training_case_ct_filenames)
    
    for x in range(0,len(training_case_ct_filenames)):  
        
        # compute mask and cropped image, save in temp_dir
        moving_mask_filename= training_case_ct_filenames[x].split('.')[0]+\
            "_similarityMask."+training_case_ct_filenames[x].split('.')[1]   
 


        moving_CT= training_case_ct_filenames[x].split('.')[0]+"_cleaned."+\
            training_case_ct_filenames[x].split('.')[1]

        fixed_CT= testing_ct_filename.split('.')[0]+"_cleaned."+\
            testing_ct_filename.split('.')[1]

        clean_ct(training_case_ct_filenames[x], moving_CT)
        clean_ct(testing_ct_filename, fixed_CT)
                        
        #compute_ct_mask_similarity(training_case_label_filenames[x],\
        #    moving_CT, moving_mask_filename)
        
        cropped_data_temp, imageInfo= nrrd.read(training_case_label_filenames[x])
        above_zero = cropped_data_temp>0
        belowthresh = cropped_data_temp<17000 #fat is 17944
    
        #threshold slice to contain only all pec data
        cropped_data_temp[above_zero & belowthresh ] = 1
        cropped_data_temp[cropped_data_temp>1] = 0  
        nrrd.write(moving_mask_filename,cropped_data_temp)
        
        compute_edge_mask(moving_mask_filename, moving_mask_filename)
                                                                   
        if (base_case_transfo == None):
            list_of_similarity_files[x] = list_of_transfos[x]+ "_measures.xml"
            transfos_for_similarity = list_of_transfos[x]
        else:
            list_of_similarity_files[x] = base_case_transfo+"_followedby_"+\
               list_of_transfos[x].split('/')[-1]+"_measures.xml"
            transfos_for_similarity = list_of_transfos[x]+","+base_case_transfo
                                            
        #similarity_call= GetTransformationSimilarityMetric+ \
        #    " --fixedCTFileName "+fixed_CT +\
        #    " --movingCTFileName "+ moving_CT+ \
        #    " --inputTransform "+transfos_for_similarity+\
        #    " --outputXMLFile "+list_of_similarity_files[x] +\
        #    " --movingLabelMapFileName "+ moving_mask_filename+\
        #    " --SimilarityMetric nc"    

        moving_CT= training_case_ct_filenames[x].split('.')[0]+"_thresholded."+\
            training_case_ct_filenames[x].split('.')[1]

        fixed_CT= testing_ct_filename.split('.')[0]+"_thresholded."+\
            testing_ct_filename.split('.')[1]
            
        GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
        "GetTransformationKappa2D")        
        similarity_call= GetTransformationSimilarityMetric+ \
            " --fixedCTFileName "+fixed_CT +\
            " --movingCTFileName "+ moving_CT+ \
            " --inputTransform "+transfos_for_similarity+\
            " --outputXMLFile "+list_of_similarity_files[x] 
            #" --movingLabelMapFileName "+ moving_mask_filename
                                                                                                        
        #print(similarity_call)                
        subprocess.call(similarity_call, shell=True);
    #print(list_of_similarity_files)     
    return list_of_similarity_files


def compute_dice_with_transfo(img_fixed, img_moving, transfo):
    
    #first transform 
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()
    temp_out = "/Users/rolaharmouche/Documents/Data/temp_reg.nrrd"        
    resamplecall = os.path.join(path['CIP_PATH'], "ResampleCT")    
        
    sys_call = resamplecall+" -d "+img_fixed+" -r "+ temp_out+\
            " -t "+transfo+" -l "+img_moving
    os.system(sys_call) 
    
    print(" computing ssd between "+img_fixed+" and registered"+ img_moving)
                    
    img1_data, info = nrrd.read(temp_out)
    img2_data, info = nrrd.read(img_fixed)
    
    
    
    #careful reference image has labels =2 and 3
    added_images = img1_data
    np.bitwise_and(img1_data, img2_data, added_images)
    Dice_calculation = sum(added_images[:])*2.0/(sum(img1_data[:])+sum(img2_data[:]))

    return Dice_calculation
    

def compute_dice_similarity_from_filenames(testing_ct_filename, \
    training_case_ct_filenames,training_case_label_filenames, list_of_transfos, \
    base_case_transfo = None):
    
    """
    generalized call to get compute similarity so that we always call the 
    same function. if  base_case_transfo = Null, then 1 transformation.
    
    make sure identity base case transfos exist.  
    
    The assumption is that base to training always exists. So we don't need 
    to invert transformations        
    
    the label filenames are supposed to be already thresholded to the right pec
    """  
    
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()
    
    print("\n\n\n In compute_dice_similarity_from_filenames \n\n\n")
    #GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
    #    "GetTransformationSimilarityMetric2D")
    GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
        "GetTransformationKappa2D")        
    list_of_similarity_files = [""]*len(training_case_ct_filenames)
    
    for x in range(0,len(training_case_ct_filenames)):  
        
        # compute mask and cropped image, save in temp_dir
        moving_mask_filename= training_case_ct_filenames[x].split('.')[0]+\
            "_similarityMask."+training_case_ct_filenames[x].split('.')[1]   
 


        #        moving_CT= training_case_ct_filenames[x].split('.')[0]+"_cleaned."+\
        #            training_case_ct_filenames[x].split('.')[1]
        #
        #        fixed_CT= testing_ct_filename.split('.')[0]+"_cleaned."+\
        #            testing_ct_filename.split('.')[1]
        #
        #        clean_ct(training_case_ct_filenames[x], moving_CT)
        #        clean_ct(testing_ct_filename, fixed_CT)
        moving_CT =training_case_ct_filenames[x]
        fixed_CT =  testing_ct_filename              
        #compute_ct_mask_similarity(training_case_label_filenames[x],\
        #    moving_CT, moving_mask_filename)
        
        cropped_data_temp, imageInfo= nrrd.read(training_case_label_filenames[x])
        above_zero = cropped_data_temp>0
        belowthresh = cropped_data_temp<17000 #fat is 17944
    
        #threshold slice to contain only all pec data
        cropped_data_temp[above_zero & belowthresh ] = 1
        cropped_data_temp[cropped_data_temp>1] = 0  
        nrrd.write(moving_mask_filename,cropped_data_temp)
        
        compute_edge_mask(moving_mask_filename, moving_mask_filename)
                                                                   
        if (base_case_transfo == None):
            list_of_similarity_files[x] = list_of_transfos[x]+ "_measures.xml"
            transfos_for_similarity = list_of_transfos[x]
        else:
            list_of_similarity_files[x] = base_case_transfo+"_followedby_"+\
               list_of_transfos[x].split('/')[-1]+"_measures.xml"
            transfos_for_similarity = list_of_transfos[x]+","+base_case_transfo
                                            
        #similarity_call= GetTransformationSimilarityMetric+ \
        #    " --fixedCTFileName "+fixed_CT +\
        #    " --movingCTFileName "+ moving_CT+ \
        #    " --inputTransform "+transfos_for_similarity+\
        #    " --outputXMLFile "+list_of_similarity_files[x] +\
        #    " --movingLabelMapFileName "+ moving_mask_filename+\
        #    " --SimilarityMetric nc"    

        temp1_for_registration = fixed_CT.split('.')[0]+"_thresholded.nrrd"
        temp2_for_registration = moving_CT.split('.')[0]+"_thresholded.nrrd"
 

        similarity_call= GetTransformationSimilarityMetric+ \
            " --fixedCTFileName "+temp1_for_registration +\
            " --movingCTFileName "+ temp2_for_registration+ \
            " --inputTransform "+transfos_for_similarity+\
            " --outputXMLFile "+list_of_similarity_files[x] 

                                              
        #print(similarity_call)                
        subprocess.call(similarity_call, shell=True);
        
        #dd = compute_dice_with_transfo(temp1_for_registration, temp2_for_registration, transfos_for_similarity)
        #list_of_similarity_files[x] = str(dd)
    #print(list_of_similarity_files)     
    return list_of_similarity_files


def crop_ct_to_moving_mask(fixed_ct_fname, moving_mask_fname, output_ct_fname, 
     is_invert, moving_to_fixed_transfo=None):
    
    """
    crops ct stored in input_ct_fname to the extent of the mask in 
    input_mask_fname. If input_transfo in not None, transforms the mask 
    first to the space of dest_ct_fname
    """
        
    toolsPaths = ['CIP_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()
            
    resamplecall = os.path.join(path['CIP_PATH'], "ResampleCT") 
    resamplecallLbl = os.path.join(path['CIP_PATH'], "ResampleLabelMap2D")       

    #create a temp file name fir the transformed mask
    temp_registered_mask = moving_mask_fname.split(".")[0]+"regtemp."+moving_mask_fname.split(".")[1]
                
    if (moving_to_fixed_transfo is None):   
        print("copying mask ")
        sys_call = ("cp "+moving_mask_fname+" "+temp_registered_mask )
    else:      
        print("transforming mask before cropping ct")
        inverse_for_similarity = " "       
        if (is_invert is True): 
            inverse_for_similarity = " -f"  
            
        # transform and save transformed MASK 
        sys_call = resamplecallLbl+" -d "+fixed_ct_fname+" -r "+ temp_registered_mask+\
            " -t "+moving_to_fixed_transfo+" -l "+moving_mask_fname+inverse_for_similarity         

    print(sys_call)
    os.system(sys_call) 
    #temp_registered_mask = moving_mask_fname 
          
    #crop the input ct to the transformed mask
    print("\n\nreading ct : "+ fixed_ct_fname)
    ct_datatemp,info1 = nrrd.read(fixed_ct_fname) # was input_ct_fname
    ct_data = np.squeeze(ct_datatemp)
    
    print("reading registered mask : "+ temp_registered_mask)    
    mask_data, info2 = nrrd.read(temp_registered_mask)
    
    print(np.shape(ct_data))
    print(np.shape(mask_data))
    assert np.shape(ct_data) == np.shape(mask_data), "CT and mask shapes are different"
    
    #ct_data[mask_data < 1] = 0
    b = np.where(mask_data>0)
    extent = [0,0]
    extent = [max(b[0])-min(b[0])+1,max(b[1])-min(b[1])+1]
    output_data = np.zeros(extent)
    output_data= ct_data[min(b[0]):max(b[0]), min(b[1]):max(b[1])]

    print("saving cropped ct output : "+ output_ct_fname)    
    print(np.shape(output_data))
    nrrd.write(output_ct_fname, output_data)  
                                                                                                                                                                                                              
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
            print (path_name + " environment variable is not set")
            exit()
    
    print("\n\n\n In register_tobase_get_closest \n\n\n")
    registerLabelMaps = os.path.join(path['CIP_PATH'],"RegisterLabelMaps2D")
    
    base_to_testing_transfo_names = [""]*len(base_case_ct_filenames)                
    for ii in range(0, len(base_case_ct_filenames)) :     
        base_case_ct_filename =  base_case_ct_filenames[ii]  
        base_to_testing_transfo_names[ii] = os.path.join(test_case_transfo_dir, \
        base_case_ct_filenames[ii].split('/')[-1].split('.')[0]+"_to_"+testing_ct_filename.split('/')[-1].split('.')[0]+"_tfm_0GenericAffine.tfm")             
        if(is_register is True):
            #register_2d_ct( base_case_ct_filename, testing_ct_filename,  base_to_testing_transfo_names[ii])
            
            temp1_for_registration = base_case_ct_filename.split('.')[0]+"_thresholded.nrrd"
            temp2_for_registration = testing_ct_filename.split('.')[0]+"_thresholded.nrrd"
            print(registerLabelMaps+" -m "+temp1_for_registration+" -f "+\
                temp2_for_registration+ " --outputTransform "+base_to_testing_transfo_names[ii])
    
            #os.system(registerLabelMaps+" -m "+base_case_ct_filename+" -f "+\
            #    testing_ct_filename+ " --outputTransform "+base_to_testing_transfo_names[ii])
        else:
            exit()        
   
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
 

    