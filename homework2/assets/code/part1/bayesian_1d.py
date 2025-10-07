import numpy as np
import math

def get_class_prob(classes, class_value):
    return np.mean(classes == class_value)

def calculate_average(classes, x1, x2, class_value):
    return np.mean(x1[(classes == class_value)]), np.mean(x2[(classes == class_value)])

def calculate_standard_deviation(classes, x1, x2, avg_x1, avg_x2, class_value):
    dev_x1 = dev_x2 = 0
    
    x1_class = x1[(classes == class_value)]
    x2_class = x2[(classes == class_value)]
    
    size = len(x1_class)
    
    for index in range(size):
        dev_x1 += (x1_class[index] - avg_x1)**2
        dev_x2 += (x2_class[index] - avg_x2)**2
        
    return dev_x1/size, dev_x2/size

def calculate_density(query, avg, dev):
    
    """
    
                    1                           (query[i] - avg) ^ 2
        ---------------------     * e ^  -  -----------------
        ( 2 * pi * dev^2 ) ^1/2                     2 * dev^2
    
    
    
    """
    dens = []
    
    size = len(query)

    for index in range(size):
        exp = (query[index] - avg[index])**2
        exp = exp / (2*dev[index]**2)
        numerator = math.exp(-exp)
        
        denominator = 2 * math.pi * dev[index]**2
        denominator = math.sqrt(denominator)
    
        density = numerator / denominator
        dens.append(density)
        
    return dens
                    
def calculate_posteriors_1d(x1,x2,classes,query,debug):

    unique = np.unique(classes)
    
    for class_index in range(len(unique)):
        
        class_value = unique[class_index]
        print("Class",class_value)
        
        prob_class = get_class_prob(classes,class_value)
        if debug: print("prob_class",prob_class)
        
        avg_x1, avg_x2 = calculate_average(classes, x1, x2, class_value)
        if debug: print("avg_x1",avg_x1,"avg_x2",avg_x2)
        
        dev_x1, dev_x2 = calculate_standard_deviation(classes, x1, x2, avg_x1, avg_x2, class_value)
        if debug: print("dev_x1",dev_x1,"dev_x2",dev_x2)
        
        dens_x1, dens_x2 = calculate_density(query, [avg_x1,avg_x2], [dev_x1,dev_x2])
        if debug: print("dens_x1",dens_x1,"dens_x2",dens_x2)
        
        score = prob_class * dens_x1 * dens_x2
        
        print("SCORE:",score)
