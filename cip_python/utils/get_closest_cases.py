import heapq
from cip_python.utils.get_mi_similarity_vec \
    import getMISimilarityVec
    
def getClosestCases(list_of_label_files, list_of_similarity_xml_files, similarity):
    """Get the closest cases to the test case using some similarity metric and
       given a list of files that contain the similarity value. 

    Parameters
    ----------
    list_of_similarity_xml_files : list of file names (string)
        Each file name points to an xml file that contains the similarity value

    list_of_label_files : list of file names (string)
        Each file name points to a file that has a labeled image
        
    similarity : string
        Type of similarity used
        ...
        
    Returns
    -------
    closest_cases : string 2D list with shape (2, num_cases)
         Contains the num_cases cases with highest similarity to the case 
         being tested, and the associated similarity value
        ...
    """

    
    num_training_cases = list_of_label_files.len
    
    #Read the similarity values 
    similarity_values = getMISimilarityVec(list_of_similarity_xml_files)
    
    #find highest matches    
    indexes=[]
    for i in range(num_training_cases):
        indexes.append(i)
    
    nlargestvalues = heapq.nlargest(11, indexes, key=lambda i: similarity_values[i]) #take (from 1 to 11)
    print(nlargestvalues)
    patient_atlas_labelmaps = [""]*num_training_cases
    patient_atlas_similarity = [1.0]*num_training_cases
    
    #Now store the num_cases in a 2D list
    for i in range(num_training_cases): 
        patient_atlas_labelmaps[i] = list_of_label_files[nlargestvalues[i+1]]
        patient_atlas_similarity[i] = similarity_values[nlargestvalues[i+1]]
    
    closest_cases = [patient_atlas_labelmaps, patient_atlas_similarity]
    return closest_cases
