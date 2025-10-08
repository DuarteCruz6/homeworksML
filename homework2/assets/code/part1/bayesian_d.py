import numpy as np
import bayesian_b as b2


def get_class_prob(classes, class_value):
    return np.mean(classes == class_value)

def get_prob_class_query(classes, x, query_value, class_value):  
    return np.sum((x == query_value) & (classes == class_value)) / np.sum((classes == class_value))

def get_densities(x1, x2, classes, query, unique):     
    densities = []
    
    for class_value in unique:
        avg_x1, avg_x2 = b2.calculate_average(classes, x1, x2, class_value)
        
        covariance_matrix = b2.calculate_covariance(classes, x1, x2, class_value, avg_x1, avg_x2)
        
        density = b2.calculate_density(covariance_matrix, query, avg_x1, avg_x2)
        
        densities.append(density)
        
    return densities
    

def calculate_most_probable(classes, x1, x2, x3, query, debug): 
    unique = np.unique(classes)
    
    densities = get_densities(x1,x2,classes,query,unique)
    
    numerators = []
    denominator = 0
    
    for class_index in range(len(unique)):
        
        class_value = unique[class_index]
        if debug:print("Class",class_value)
        
        class_prob = get_class_prob(classes, class_value)
        if debug: print("class prob",class_prob)
        
        prob_class_x3 = get_prob_class_query(classes, x3, query[2], class_value)
        if debug: print("prob_class_x3",prob_class_x3)
        
        density = densities[class_index]
        if debug: print("density",density)
        
        prod = class_prob*prob_class_x3*density
        numerators.append(prod)
        denominator+= prod
        
    for class_index in range(len(unique)):   
        print("Class",unique[class_index])
        if debug: print("numerator:", numerators[class_index])
        if debug: print("denominator:",numerators[0],"+",numerators[1])
        res = numerators[class_index]/denominator
        print("RESULT:",res)
        
        
        

        
        
        
        
        