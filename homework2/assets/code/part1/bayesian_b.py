import numpy as np
import math

def get_class_prob(classes, class_value):
    return np.mean(classes == class_value)

        
def calculate_average(classes, x1, x2, class_value):
    return np.mean(x1[(classes == class_value)]), np.mean(x2[(classes == class_value)])


def calculate_covariance(classes, x1, x2, class_value, avg_x1, avg_x2):
    matrix = []
    
    value0 = value1 = value2 = 0    ### [ value0 value 1]
                                    ### [ value1 value 2]
    
    x1_class = x1[(classes == class_value)]
    x2_class = x2[(classes == class_value)]
    
    size = len(x1_class)
    
    for index in range(size):
        
        x1_value = x1_class[index] - avg_x1
        x2_value = x2_class[index] - avg_x2
        
        value0 += x1_value**2
        value1 += x1_value * x2_value
        value2 += x2_value**2
        
    matrix.append([value0/(size - 1), value1/(size - 1)])
    matrix.append([value1/(size - 1), value2/(size - 1)])
    
    return np.array(matrix)

def invert_matrix(determ, covariance_matrix):
    inv_determ = 1 / determ
    a = covariance_matrix[0][0]
    b = covariance_matrix[0][1]
    c = covariance_matrix[1][1]
    
    inv = np.zeros((2,2))
    inv[0][0] =  c * inv_determ
    inv[1][1] =  a * inv_determ
    inv[0][1] = -b * inv_determ
    inv[1][0] = -b * inv_determ
    
    return inv

    
    
def matrix_multiply(covariance_matrix, query, avg_x1, avg_x2):
    
    a = query[0] - avg_x1
    b = query[1] - avg_x2
    
    c = covariance_matrix[0][0]
    d = covariance_matrix[0][1]
    e = covariance_matrix[1][1]
    
    mul = c*a*a + 2*d*a*b + e*b*b

    return mul


def calculate_density(covariance_matrix, query, avg_x1, avg_x2):
    
    determ = (covariance_matrix[0][0]*covariance_matrix[1][1] - covariance_matrix[0][1]**2)
    
    inv_cov = invert_matrix(determ, covariance_matrix.copy())

    exp_term = -0.5 * matrix_multiply(inv_cov, query, avg_x1, avg_x2)
    
    numerator = math.exp(exp_term)
    denominator = 2 * math.pi * math.sqrt(determ)
    
    return numerator / denominator
    
        
def calculate_posteriors_2d(x1,x2,classes,query, debug):      
    unique = np.unique(classes)
    
    numerators = []
    denominator = 0
    
    for class_index in range(len(unique)):
        
        class_value = unique[class_index]
        if debug: print("Class",class_value)
        
        avg_x1, avg_x2 = calculate_average(classes, x1, x2, class_value)
        if debug: print("avg:", avg_x1, avg_x2)
        
        covariance_matrix = calculate_covariance(classes, x1, x2, class_value, avg_x1, avg_x2)
        if debug: print("matrix")
        if debug: print(covariance_matrix)
        
        density = calculate_density(covariance_matrix, query, avg_x1, avg_x2) 
        if debug: print("density:", density)
        
        numerator = get_class_prob(classes,class_value) * density
        numerators.append(numerator)
        
        denominator+= numerator
        
    results = []
        
    for class_index in range(len(unique)):   
        print("Class",unique[class_index])
        if debug: print("numerator:", numerators[class_index])
        if debug: print("denominator:",numerators[0],"+",numerators[1])
        res = numerators[class_index]/denominator
        results.append(res)
        print("RESULT:",res)
        
    return results
