import os
from lxml import etree
import nrrd
import numpy as np
from scipy import ndimage

def compute_image_gradient(input_filename, output_filename):
    """
    compute the gradient of an image using unu tools
    """
    toolsPaths = ['CIP_PATH', 'TEEM_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()  

    unuCall = os.path.join(path['TEEM_PATH'], \
            "unu")
    vprobe_call = os.path.join(path['TEEM_PATH'], \
            "vprobe")
            
    #gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+input_mask_name+" | \
    #        vprobe -i - -k scalar -q gmag -o "+output_mask_name     
    #os.system(gradient_call);             
    
    print("computing image gradient with image "+input_filename+"\n")
    
    gradient_call = unuCall+" join -i "+input_filename+" "+ input_filename+\
        " -a 2 | "+unuCall+" crop -min 0 0 0 -max M M 0 | "+unuCall+" axinfo -a 0 1 2 -sp 1 -k domain | "+unuCall+\
        " resample -k dg:0.7,3 -s x1 x1 x1 -i - | "+vprobe_call+" -i - -k scalar -q gmag -o "+output_filename  
    print(gradient_call)
    os.system(gradient_call) 


def blur_image(input_filename, sigma, output_filename):
    """
    blur an image using a soecific sigma
    """ 
    toolsPaths = ['CIP_PATH', 'TEEM_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()  
            

    unuCall = os.path.join(path['TEEM_PATH'], \
            "unu")   
            
    image, info = nrrd.read(input_filename)     
    if(info['dimension'] == 3):   
        blur_call = unuCall+" resample -s x1 x1 x1 -k gauss:"+str(sigma)+","+str(sigma)+" -i "+\
            input_filename+" -o "+ output_filename
    else:
        blur_call = unuCall+" resample -s x1 x1 -k gauss:"+str(sigma)+","+str(sigma)+" -i "+\
            input_filename+" -o "+ output_filename
    print(blur_call)
    os.system(blur_call)     
   
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

def compute_simple_mask(input_mask_name, output_mask_name):
    """
    generates a mask with some dilation of the input mask
    """

    mask_data, options = nrrd.read(input_mask_name)
    print(" reading mask name "+input_mask_name)
    #mask_data = np.array(ndimage.binary_dilation(mask_data, iterations = 10)).astype(np.int) 
    nrrd.write(output_mask_name,mask_data)

def compute_image_similarity(fixed_image, registered_moving_image, image_feature, similarity, mask_label_value, the_smoothing_param, registered_mask = None):
    """
    Compute the similarity value between the fixed and the registered moving image given 
    a specific image feature and a similarity measure. Assume a mask that covers only the region of
    interest (with no dilation).
    """
    
    os.environ['ITKTOOLS_PATH']=os.path.expanduser('/Users/rolaharmouche/Downloads/ITKTools/build/bin')
    os.environ['TEEM_PATH']=os.path.expanduser('/Users/rolaharmouche/ChestImagingPlatformPrivate-build/teem-build/bin/')
    os.environ['ANTS_PATH']=os.path.expanduser('/Users/rolaharmouche/Downloads/antsbin/bin')
    os.environ['CIP_PATH']=os.path.expanduser('/Users/rolaharmouche/ChestImagingPlatformPrivate-build/CIP-build/bin')

    toolsPaths = ['CIP_PATH', 'TEEM_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print (path_name + " environment variable is not set")
            exit()    

    possible_similarities = ["nc", "mi", "msqr", "gd", "nmi"]
    possible_features = ["intensity", "gradient", "intensity_around_edge", "gradient_around_edge", "smoothed_gradient_around_edge"]
    GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
            "GetTransformationSimilarityMetric2D")
    unuCall = os.path.join(path['TEEM_PATH'], \
            "unu")

    try:
        possible_similarities.index(similarity)        
    except ValueError:
        print("unknown similarity")        

    """
    prepare the input images depending on image feature
    """
    try:
        possible_similarities.index(similarity)        
    except ValueError:
        print("unknown similarity, should be one of :") 
        print(possible_similarities)
    try:
        possible_features.index(image_feature)        
    except ValueError:
        print("unknown feature")   
    print("pec label to check "+str(mask_label_value))          

    fixed_id = "_".join(fixed_image.split("/")[-1].split("_")[0:5])
    moving_id = "_".join(registered_moving_image.split("/")[-1].split("_")[0:5]) 
    similarity_file = moving_id+"_tmp.xml"         
    mask_for_similarity = moving_id+"_temp_similarity_mask.nrrd"
    
    mask_data, mas_info = nrrd.read(registered_mask)
    mask_data[mask_data == mask_label_value] = 1

    mask_data[mask_data > 1] = 0
    #mask_data2 = np.zeros_like(mask_data)       
    nrrd.write(mask_for_similarity, mask_data)

    if image_feature is "intensity":
        print("Computing similarity based on intensities");
        moving_image_forsimilarity = registered_moving_image
        fixed_image_forsimilarity = fixed_image
        if (the_smoothing_param > 0):
            fixed_image_forsimilarity = moving_id+"_fixed_image_blurred.nrrd"
            moving_image_forsimilarity = moving_id+"_moving_image_blurred.nrrd"
            blur_image(fixed_image, the_smoothing_param, fixed_image_forsimilarity)
            blur_image(registered_moving_image, the_smoothing_param, moving_image_forsimilarity) 
        compute_simple_mask(mask_for_similarity, mask_for_similarity)

    elif image_feature is "gradient":
        print("Computing similarity based on gradient");
        fixed_image_forsimilarity = moving_id+"_fixed_image_gradient.nrrd"
        moving_image_forsimilarity = moving_id+"_moving_image_gradient.nrrd"
        compute_image_gradient(fixed_image, fixed_image_forsimilarity)
        compute_image_gradient(registered_moving_image, moving_image_forsimilarity)
    elif image_feature is "intensity_around_edge":
        print("Computing similarity based on intensities around the edge of the pec");
        try:
            len(registered_mask)        
        except ValueError:
            print("mask cannot be None")
        moving_image_forsimilarity = registered_moving_image
        fixed_image_forsimilarity = fixed_image
        if (the_smoothing_param > 0):
            fixed_image_forsimilarity = moving_id+"_fixed_image_blurred.nrrd"
            moving_image_forsimilarity = moving_id+"_moving_image_blurred.nrrd"
            blur_image(fixed_image, the_smoothing_param, fixed_image_forsimilarity)
            blur_image(registered_moving_image, the_smoothing_param, moving_image_forsimilarity) 
               
        compute_edge_mask(mask_for_similarity, mask_for_similarity)        
    elif image_feature is "gradient_around_edge":
        print("Computing similarity based on gradient around the edge of the pec");
        try:
            len(registered_mask)        
        except ValueError:
            print("mask cannot be None")
        fixed_image_forsimilarity = moving_id+"_fixed_image_gradient.nrrd"
        moving_image_forsimilarity = moving_id+"_moving_image_gradient.nrrd"
        
        compute_image_gradient(fixed_image, fixed_image_forsimilarity)
        compute_image_gradient(registered_moving_image, moving_image_forsimilarity)
        
        
        compute_edge_mask(mask_for_similarity, mask_for_similarity)       
    elif image_feature is "smoothed_gradient_around_edge":
        print("Computing similarity based on blurred gradient around the edge of the pec");
        try:
            len(registered_mask)        
        except ValueError:
            print("mask cannot be None")
        fixed_image_forsimilarity = moving_id+"_fixed_image_gradient.nrrd"
        moving_image_forsimilarity = moving_id+"_moving_image_gradient.nrrd"
                
        blur_image(fixed_image, the_smoothing_param, fixed_image_forsimilarity)
        blur_image(registered_moving_image, the_smoothing_param, moving_image_forsimilarity) 
        compute_image_gradient(fixed_image, fixed_image_forsimilarity)
        compute_image_gradient(registered_moving_image, moving_image_forsimilarity)   
        blur_image(fixed_image_forsimilarity, the_smoothing_param, fixed_image_forsimilarity)
        blur_image(moving_image_forsimilarity, the_smoothing_param, moving_image_forsimilarity)     
#unu join -i 10087S_INSP_STD_NJC_COPD_to_10004O_INSP_STD_BWH_COPD_tfm_0GenericAffine_pecsSubqFat.tfm_pecSlice.nrrd 10087S_INSP_STD_NJC_COPD_to_10004O_INSP_STD_BWH_COPD_tfm_0GenericAffine_pecsSubqFat.tfm_pecSlice.nrrd -a 2 | unu crop -min 0 0 0 -max M M 0 | unu axinfo -a 0 1 2 -sp 1 -k domain | unu resample -k dg:0.7,3 -s x1 x1 x1 -i - -o to_probe.nrrd
#~/teem-code/teem-build/bin/vprobe -k scalar -q hess -i to_probe.nrrd -o kk.nrrd
#gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+registered_moving_image+" | \
#            vprobe -i - -k scalar -q gmag -o "+moving_image_forsimilarity 
        #print(gradient_call)
        #os.system(gradient_call);    
        #gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+fixed_image+" | \
        #    vprobe -i - -k scalar -q gmag -o "+fixed_image_forsimilarity 
        #os.system(gradient_call); 
        mask_for_similarity = moving_id+"_temp_similarity_mask.nrrd"
        compute_edge_mask(mask_for_similarity, mask_for_similarity)        
                                                
    similarity_call= GetTransformationSimilarityMetric+ \
            " --fixedCTFileName "+fixed_image_forsimilarity +\
            " --movingCTFileName "+ moving_image_forsimilarity+ \
            " --outputXMLFile "+similarity_file +\
            " --SimilarityMetric "+similarity          
    if(registered_mask != None):
         similarity_call=similarity_call+" --fixedLabelMapFileName "+ mask_for_similarity
    
    os.system("rm "+similarity_file)          
    print(similarity_call)     
    os.system(similarity_call);
    
    try:
        tree = etree.parse(similarity_file)
        similarity_val = float(tree.find('SimilarityValue').text)
    except IOError:
        print("similarity file not computed")
        similarity_val = 0
        
    return similarity_val