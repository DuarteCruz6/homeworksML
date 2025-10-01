import numpy as np

def get_class_prob(classes, class_value):
    return np.mean(classes == class_value)


def get_query_class(classes, class_value, x1, x2, query):
    total_class = np.sum((classes == class_value))
    
    if total_class == 0:
        return 0.0, 0.0
    
    query_0_class = np.sum(x1[(classes == class_value)] == query[0])
    query_1_class = np.sum(x2[(classes == class_value)] == query[1])
    
    return query_0_class / total_class, query_1_class / total_class


def get_prob_query(x1, x2, query):
    prob_query_0 = np.mean(x1 == query[0])
    prob_query_1 = np.mean(x2 == query[1])
    
    return prob_query_0, prob_query_1
          
                    
def calculate_posteriors_1d(x1,x2,classes,query,debug):
    """
                    P(C=A) * P(X1=query_vector[0] | C=A) * P(X2=query_vector[1] | C=A)
    posterior_A = -------------------------------------------------------------------------
                            P(X1=query_vector[0]) * P(X2=query_vector[1])
    
    """
    
    unique = np.unique(classes)
    
    prob_query_0,  prob_query_1 = get_prob_query(x1,x2,query)
    
    for class_index in range(len(unique)):
        
        class_value = unique[class_index]
        
        print("Class",class_value)
        
        prob_class = get_class_prob(classes, class_value)
        prob_x1, prob_x2 = get_query_class(classes, class_value, x1, x2, query)
        
        numerator = prob_class * prob_x1 * prob_x2
        
        denominator = prob_query_0*prob_query_1
        
        if debug: print("numerator:",prob_class,"*",prob_x1,"*",prob_x2)
        if debug: print("denominator:",prob_query_0,"*",prob_query_1)
        
        print("RESULT:",numerator/denominator) 