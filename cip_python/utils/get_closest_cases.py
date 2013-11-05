def get_closest_cases(input_image, list_of_label_files):
    """Get the closest cases to the test case using some
       similarity metric. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    list_of_label_files : list of file names (string)
        Each file name points to a file that has a labeled 
        ...
        
    Returns
    -------
    closest_cases : 2D array with shape (2, num_cases)
         The num_cases cases with highest similarity to the case 
         being tested
        ...
    """

    ###Find highest 10 matches similarity-wise for that patient
    patient_row = lung_segmentation_helper.getSimilarityVectorFromMI(patient_names, CaseID, data_dir+study+'/',transfo_dir,xml_generate)
    
    indexes=[]
    for i in range(100):
        indexes.append(i)
    
    nlargestvalues = heapq.nlargest(11, indexes, key=lambda i: patient_row[i]) #take (from 1 to 11)
    print(nlargestvalues)
    patient_atlas_caselist = [""]*num_cases
    patient_atlas_similarity = [1.0]*num_cases
    for i in range(num_cases): 
        patient_atlas_caselist[i] = patient_names[nlargestvalues[i+1]]
        patient_atlas_similarity[i] = patient_row[nlargestvalues[i+1]]
    
    closest_cases = [patient_atlas_caselist patient_atlas_similarity]
    return closest_cases
