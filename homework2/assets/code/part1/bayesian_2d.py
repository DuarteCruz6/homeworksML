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
        
    matrix.append([value0/size, value1/size])
    matrix.append([value1/size, value2/size])
    
    return np.array(matrix)


def invert_matrix(determ, covariance_matrix):
    
    ### [value0 value1]
    ### [value1 value2]
    
    determ = 1/determ 
    a = covariance_matrix[0][0]
    covariance_matrix[0][0] = covariance_matrix[1][1] * determ
    covariance_matrix[1][1] = a * determ
    covariance_matrix[0][1] *= -determ
    covariance_matrix[1][0] = covariance_matrix[0][1]
    
    return covariance_matrix
    
    
def matrix_multiply(covariance_matrix, query, avg_x1, avg_x2):
    
    a = query[0] - avg_x1
    b = query[1] - avg_x2
    
    c = covariance_matrix[0][0]
    d = covariance_matrix[0][1]
    e = covariance_matrix[1][0]
    f = covariance_matrix[1][1]
    
    mul = c * (a**2)
    mul += a * b * e
    mul += a * b * d
    mul += f * (b**2)
    
    return mul
    
    
def calculate_density(covariance_matrix, query, avg_x1, avg_x2):

    determ = (covariance_matrix[0][0]*covariance_matrix[1][1] - covariance_matrix[0][1]**2)
    
    covariance_matrix = invert_matrix(determ, covariance_matrix)

    exp = -0.5 * matrix_multiply(covariance_matrix, query, avg_x1, avg_x2)

    numerator = math.exp(exp)
    denominator = 2 * math.pi * (determ ** 0.5)
    
    return numerator/denominator
    
        
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
        
    for class_index in range(len(unique)):   
        print("Class",unique[class_index])
        if debug: print("numerator:", numerators[class_index])
        if debug: print("denominator:",numerators[0],"+",numerators[1])
        print("RESULT:",numerators[class_index]/denominator)
