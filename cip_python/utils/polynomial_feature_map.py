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
      self.list_of_orders = input_orders
      self.feature_vectors = input_features
      self.num_terms = -1
      self.num_terms_per_order = np.zeros(np.shape(input_features)[0])
    
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
        m = len(self.feature_vectors)
        f = math.factorial
        for x in range(0, len(self.list_of_orders)):
            n = self.list_of_orders[x]
            self.num_terms_per_order[x] = math.factorial(n+m-1) /( math.factorial(n) * math.factorial(m-1))
            b = self.num_terms_per_order[x]
            print("order="+str(x)+"numterms="+str(b))
            self.num_terms=self.num_terms+self.num_terms_per_order[x]
        
    def get_mapped_feature_vectors(self):
        """
        get all the mapped feature vectors following polynomial expansion
        """
        pass
        
    def get_mapped_feature_vector_element(self,element_index):
        """
        get the mapped feature vector at a particular element 
        """
        
        #first find the order and location of the particular element
        order_index = 0
        cumul_numterms = self.num_terms_per_order[order_index]
        index_within_order= element_index
        while self.num_terms_per_order[order_index] <= (index_within_order):  
            index_within_order = index_within_order-self.num_terms_per_order[order_index]
            order_index=order_index+1        
            cumul_numterms+=self.num_terms_per_order[order_index]
            
        print("order_index="+str(order_index))
        print("element within order = "+str(index_within_order))
        #now now through a case statement for the established orders and terms
        if self.list_of_orders[order_index] is 1:
            return self.feature_vectors[element_index]
        if self.list_of_orders[order_index] is 2:
            if index_within_order is 0:
                return self.feature_vectors[0]*self.feature_vectors[0]
            if index_within_order is 1:
                return self.feature_vectors[0]*self.feature_vectors[1]
            if index_within_order is 2:
                return self.feature_vectors[1]*self.feature_vectors[1]        