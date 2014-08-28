import os
from lxml import etree
import nrrd
import numpy as np
from scipy import ndimage

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
            print path_name + " environment variable is not set"
            exit()    

    unuCall = os.path.join(path['TEEM_PATH'], \
            "unu")
    gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+input_mask_name+" | \
            vprobe -i - -k scalar -q gmag -o "+output_mask_name     
    os.system(gradient_call); 
    mask_data, options = nrrd.read(output_mask_name)
    dilated_mask = np.array(ndimage.binary_dilation(mask_data, iterations = 20)).astype(np.int) 
    erdoded_mask = np.array(ndimage.binary_erosion(mask_data, iterations = 20)).astype(np.int) 
    
    final_array = np.zeros_like(dilated_mask)
    final_array = np.bitwise_and(dilated_mask, np.bitwise_invert(erdoded_mask))
    nrrd.write(output_mask_name,final_array)
    #to redo : compute distance map and threshold
    #    call(convertCall+imageDir+imageName+".nrrd -t uchar -o "+imageDir+imageName+"_uchar.nrrd")
    #find distance from ribs
#    call(ITKToolsDir+pxdistancetransformCall+imageDir+imageName+"_uchar.nrrd -out "+imageDir+imageName+"Distance.nrrd")

def compute_image_similarity(fixed_image, registered_moving_image, image_feature, similarity, registered_mask = None):
    """
    Compute the similarity value between the fixed and the registered moving image given 
    a specific image feature and a similarity measure
    
    
    """
    os.environ['ITKTOOLS_PATH']=os.path.expanduser('/Users/rolaharmouche/Downloads/ITKTools/build/bin')
    os.environ['TEEM_PATH']=os.path.expanduser('/Users/rolaharmouche/Slicer4-SuperBuild-Debug/teem-build/bin')
    os.environ['ANTS_PATH']=os.path.expanduser('/Users/rolaharmouche/Downloads/antsbin/bin')
    os.environ['CIP_PATH']=os.path.expanduser('/Users/rolaharmouche/ChestImagingPlatformPrivate-build/CIP-build/bin')

    toolsPaths = ['CIP_PATH', 'TEEM_PATH'];
    path=dict()
    for path_name in toolsPaths:
        path[path_name]=os.environ.get(path_name,False)
        if path[path_name] == False:
            print path_name + " environment variable is not set"
            exit()    

    possible_similarities = ["nc", "mi", "msqr", "gd", "nmi"]
    possible_features = ["intensity", "gradient", "intensity_around_edge", "gradient_around_edge"]
    GetTransformationSimilarityMetric = os.path.join(path['CIP_PATH'], \
            "GetTransformationSimilarityMetric2D")
    unuCall = os.path.join(path['TEEM_PATH'], \
            "unu")
    similarity_file = "tmp.xml"
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
        
    if image_feature is "intensity":
            print("Computing similarity based on intensities");
            moving_image_forsimilarity = registered_moving_image
            fixed_image_forsimilarity = fixed_image
            mask_for_similarity = registered_mask
    elif image_feature is "gradient":
        print("Computing similarity based on gradient");
        fixed_image_forsimilarity = "fixed_image_gradient.nrrd"
        moving_image_forsimilarity = "moving_image_gradient.nrrd"
        gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+registered_moving_image+" | \
            vprobe -i - -k scalar -q gmag -o "+moving_image_forsimilarity 
        os.system(gradient_call);    
        gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+fixed_image+" | \
            vprobe -i - -k scalar -q gmag -o "+fixed_image_forsimilarity 
        os.system(gradient_call); 
        mask_for_similarity = registered_mask
    elif image_feature is "intensity_around_edge":
        print("Computing similarity based on intensities around the edge of the pec");
        try:
            len(registered_mask)        
        except ValueError:
            print("mask cannot be None")
        moving_image_forsimilarity = registered_moving_image
        fixed_image_forsimilarity = fixed_image
        mask_for_similarity = "temp_similarity_mask.nrrd"
        compute_edge_mask(registered_mask, mask_for_similarity)        
    elif image_feature is "gradient_around_edge":
        print("Computing similarity based on gradient around the edge of the pec");
        try:
            len(registered_mask)        
        except ValueError:
            print("mask cannot be None")
        fixed_image_forsimilarity = "fixed_image_gradient.nrrd"
        moving_image_forsimilarity = "moving_image_gradient.nrrd"
        gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+registered_moving_image+" | \
            vprobe -i - -k scalar -q gmag -o "+moving_image_forsimilarity 
        os.system(gradient_call);    
        gradient_call = unuCall+" resample -k dg:0.7,3 -s x1 x1 x1 -i "+fixed_image+" | \
            vprobe -i - -k scalar -q gmag -o "+fixed_image_forsimilarity 
        os.system(gradient_call); 
        mask_for_similarity = "temp_similarity_mask.nrrd"
        compute_edge_mask(registered_mask, mask_for_similarity)
        
                                                
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
        
    return similarity_val