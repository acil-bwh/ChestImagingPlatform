import pdb
import math
import numpy as np

class polynomial_feature_map:
    
    """class for computing polynomial elements given a feature vector

    Parameters
    ----------
    feature_vectors: list of arrays
        The input feature vectors for which the polynomial needs to be computed
    orders: list of interegrs 
    num_terms: number of terms for the mapped feature vector
        
    """
    def __init__(self, input_orders, input_features):
      list_of_orders = input_orders
      feature_vectors = input_features
      num_terms = -1
      num_terms_per_order = np.zeros(input_features.len)
      
    """function that computes the number of terms in the feature map given the list
     of orders 
    ----------
    return: none 
        
    """
    def compute_num_terms(self):
        """
        go through list of orders and add corresponding numterms
        """
        #http://mathforum.org/library/drmath/view/68607.html. m terms to the nth power:
        #(n+(m-1))!
        #---------- = "(n+m-1) choose (m-1)"
        #(n!)(m-1)!
        self.num_terms=0
        m = self.input_features.len
        f = math.factorial
        for x in range(0, self.list_of_orders.len):
            n = self.list_of_orders[x]
            self.num_terms_per_order[x] = f(n+m-1) /( f(n) * f(m-1))
            self.num_terms+=self.num_terms_per_order[x]
        
    def get_mapped_feature_vectors(self, list_of_features):
        """
        get all the mapped feature vectors following polynomial expansion
        """
        pass
        
    def get_mapped_feature_vector_element(self,element_index, list_of_features):
        """
        get the mapped feature vector at a particular element 
        """
        
        #first find the order and location of the particular element
        order_index = 0
        cumul_numterms = self.num_terms_per_order[order_index]
        element_within_order = element_index
        while cumul_numterms < (element_index+1):          
            order_index=order_index+1
            cumul_numterms+=self.num_terms_per_order[order_index]
            element_within_order=element_within_order-self.num_terms_per_order[order_index]
            
        #now now through a case statement for the established orders and terms
        if self.list_of_orders[order_index] is 1:
            return list_of_features[element_index]
        if self.list_of_orders[order_index] is 2:
            if element_within_order is 0:
                return list_of_features[0]*list_of_features[0]
            if element_within_order is 1:
                return list_of_features[0]*list_of_features[1]
            if element_within_order is 2:
                return list_of_features[1]*list_of_features[1]        