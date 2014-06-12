import numpy as np
from lxml import etree

#def getMISimilarityVec(input_image, list_of_image_files, \
#    list_of_tranformation_files):
#    """Obtain a similarity vector which contains similarity values between a 
#         a source case and a list of target cases. Assumes a transformation \
#         file exists.
#        
#
#    Parameters
#    ----------
#    input_image : integer array, shape (L, M, N)
#   
#    list_of_image_files : list of strings,
#        contains the image file names. The similarity between each of the images
#        in the list and the input_image is to be computed. 
#    
#    list_of_tranformation_files : list of strings, 
#        contains the transformation files to be applied to the corresponding \
#        image in the list_of_image_files
#
#        ...
#        
#    Returns
#    -------
#    similatity_vec : list of floats
#        Similarity values between the input_image and the associated images \
#        in list_of_image_files (at the same location in the array)
#    """

def getMISimilarityVec(list_of_similarity_files):
    """Obtain a similarity vector which contains similarity values between a 
         a source case and a list of target cases. Assumes a similarity \
         file exists.
        

    Parameters
    ----------

    list_of_similarity_files : list of strings, 
        contains the similarity files to be searched in order to find the \
        similarity value

        ...
        
    Returns
    -------
    similatity_vec : list of floats
        Similarity values between the input_image and the associated images \
        in list_of_image_files (at the same location in the array)
    """

    
    num_images = len(list_of_similarity_files)
    
    #initialize an array of the same length as the number of files
    similarity_vec = [-1000.0]*num_images
    
    #loop through all files and find similarity
    for i in range(num_images): 
        #Read the xml file and extract the similarity value
        tree = etree.parse(list_of_similarity_files[i])
        similarity_vec[i] = float(tree.find('SimilarityValue').text)
    return similarity_vec