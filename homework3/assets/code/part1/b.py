import numpy as np

def calculate_left(design_matrix):
    transposed = np.transpose(design_matrix)
    return transposed @ design_matrix # does matrix multiplication

def calculate_right(design_matrix, targets):
    transposed = np.transpose(design_matrix)
    return transposed @ targets # does matrix multiplication


def calculate_regression_weights(design_matrix, targets, debug):
    """
      
    ΦT Φ w = ΦT t <=> A w = B

    ΦT = Φ transposed
    w = weights
    t = targets
    
    """

    a = calculate_left(design_matrix) # returns ΦT Φ
    if debug: print("ΦT Φ =\n", a)
    
    b = calculate_right(design_matrix, targets) # returns ΦT t
    if debug: print("ΦT t =\n", b.reshape(-1, 1))

    w = np.linalg.pinv(a) @ b # does A-1 B
    w = w.reshape(-1, 1)
    print("RESULT:\n",w)