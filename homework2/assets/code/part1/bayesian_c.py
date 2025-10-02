import numpy  as np

def get_total_class_query(classes, x3, query, class_value):    
    return np.sum((x3 == query[0]) & (classes == class_value))

def calculate_most_probable(classes, x3, query, debug):
    
    unique = np.unique(classes)
    
    query_total = np.sum((x3 == query[0]))
    
    max_prob = 0
    max_prob_class_value = ""
    
    for class_index in range(len(unique)):
        
        class_value = unique[class_index]
        print("Class",class_value)
        
        class_total = get_total_class_query(classes, x3, query, class_value)
        
        res = class_total/query_total
        
        if debug: print("Numerator:",class_total)
        if debug: print("Denominator:",query_total)
        
        
        print("RESULT:", res)
        
        if res>max_prob: max_prob = res; max_prob_class_value = class_value
    
    print("MOST PROBABLE CLASS:",max_prob_class_value,"with prob of",max_prob)
        
    
    