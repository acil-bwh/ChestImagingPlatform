import heapq
num_cases=10

def construct_lung_atlas(input_image, list_of_label_files):
    """Construct the lung atlas using training labeled data. 

    Parameters
    ----------
    input_image : float array, shape (L, M, N)

    list_of_label_files : list of file names (string)
        Each file name points to a file that has a labeled 
        ...
        
    Returns
    -------
    priors : list of float arrays with shape (L, M, N)
        Each of left lung, right lung, not lung will be represented by an array 
        having the same size as the input image. Every voxel must have a value 
        in the interval [0, 1], indicating the probability of that particular
        structure being present at that particular location.
        ...
    """
    closest_cases = get_closest_cases(input_image, list_of_label_files)
    
    priors = generate_atlases_given_caselist(input_image, closest_cases)
    
    return priors
    
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
    
    
def getSimilarityVectorFromMI(patient_names, case_id, data_dir,transfo_dir,xml_generate):
    mat_dim=patient_names.len    
    sim_mat = np.ones(mat_dim)
    sim_mat = sim_mat*(-1000.0)
    
    #loop through all files and find similarity
    
    return sim_mat

def generate_atlases_given_caselist(nput_image, closest_cases):
    
