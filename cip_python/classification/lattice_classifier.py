class kde_lattice_classifier:
    """General purpose class implementing a lattice kernel density classification. 

    The user inputs a list of dataframe with kde, distance and neighbourhood 
    information (one df per patch). Each patch  is segmented into one of the 
    predefined classes. The output is a dataframe with a class label for patch 
    ID.
       
    Example scikit learn code with a graph: label_propagation.py   
    Parameters 
    ----------    
    kde_lower_limit:int
        Lower limit of kde histogram
        
    kde_upper_limit:int
        Upper limit of kde histogram
        
    Attribues
    ---------
    _patch_labels_df : pandas dataframe
        Contains the class labels corresponding to the input patch ids
    """
    
    def __init__(self, num_knn_neighbours):
        pass
        
    def fit(self, input_kde, input_distance, input_neighbours):
        pass
    
    def predict(self):
        pass
                
    def get_patch_unaries(self):
        pass
        