import heapq
import numpy as np
from cip_python.utils.get_mi_similarity_vec \
    import getMISimilarityVec
    
def getClosestCases(list_of_label_files, list_of_similarity_xml_files, \
    similarity, num_closest_cases):
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
        
    num_closest_cases : int
        The number of closest cases to return
        ...
        
    Returns
    -------
    closest_cases : string 2D list with shape (2, num_cases)
         Contains the num_cases cases with highest similarity to the case 
         being tested, and the associated similarity value
        ...
    """

    #TODO: Account for different similarities
    
    num_training_cases = len(list_of_label_files)
    
    #Read the similarity values 
    similarity_values = getMISimilarityVec(list_of_similarity_xml_files)
    print(similarity_values)

    #find highest matches    
    indexes=[]
    for i in range(num_training_cases):
        indexes.append(i)
    if (similarity == "NCC"):
        nlargestvalues = heapq.nlargest(num_closest_cases, indexes, key=lambda \
            i: abs(float(similarity_values[i]))) #take (from 0 to 10, assuming testint is not in trainng)
    else:
        nlargestvalues = heapq.nlargest(num_closest_cases, indexes, key=lambda \
            i: (similarity_values[i])) #take (from 0 to 10, assuming testint is not in trainng)        
    print(nlargestvalues)
    patient_atlas_labelmaps = [""]*num_closest_cases
    patient_atlas_similarity = [1.0]*num_closest_cases
    
    #Now store the num_cases in a 2D list
    for i in range(num_closest_cases): 
        patient_atlas_labelmaps[i] = list_of_label_files[nlargestvalues[i]]
        print(patient_atlas_labelmaps[i])
        patient_atlas_similarity[i] = similarity_values[nlargestvalues[i]]
    
    closest_cases = np.vstack((patient_atlas_labelmaps, patient_atlas_similarity))
    return closest_cases

def getRandomCases(list_of_label_files, num_closest_cases):  
    
    #find 10 random patients from the list (make sure that they exclude the patient

    patient_atlas_labelmaps = [""]*num_closest_cases
    patient_atlas_similarity = [0.1]*num_closest_cases
    nrandomvalues= np.random.randint(0, 99, 15)
    
    nrandomvalues3 = np.unique(nrandomvalues)[0:10]
    for i in range(num_closest_cases): #i=0,9
        patient_atlas_labelmaps[i] = list_of_label_files[nrandomvalues3[i]]
        
    random_cases = np.vstack((patient_atlas_labelmaps, patient_atlas_similarity))
    return random_cases