import numpy as np

def get_mi_similarity_vec(patient_names, case_id, data_dir, transfo_dir,
                          xml_generate):
    """
    """
    mat_dim=patient_names.len    
    sim_mat = np.ones(mat_dim)
    sim_mat = sim_mat*(-1000.0)
    
    #loop through all files and find similarity
    
    return sim_mat