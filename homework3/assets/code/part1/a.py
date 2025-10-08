import numpy as np 

def calculate_design_matrix(training_set, size):
    matrix = []

    for elem in training_set:
        row = []
        for exp in range(size+1):
            row.append(elem**exp)
        
        matrix.append(np.array(row))

    return np.array(matrix)