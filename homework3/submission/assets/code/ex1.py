import numpy as np

def compute_normal_matrix(design_matrix):
    transposed = np.transpose(design_matrix)
    return transposed @ design_matrix # does matrix multiplication

def compute_target_projection(design_matrix, targets):
    transposed = np.transpose(design_matrix)
    return transposed @ targets # does matrix multiplication


def calculate_regression_weights(design_matrix, targets):
    """
      
    ΦT Φ w = ΦT t <=> A w = B

    ΦT = Φ transposed
    w = weights
    t = targets
    
    """

    a = compute_normal_matrix(design_matrix) # returns ΦT Φ
    print("ΦT Φ =\n", a)
    
    b = compute_target_projection(design_matrix, targets) # returns ΦT t
    print("ΦT t =\n", b.reshape(-1, 1))

    w = np.linalg.pinv(a) @ b # does A-1 B
    w = w.reshape(-1, 1)
    print("RESULT:\n",w)
    
    
    
# data provided in the question
t1 = -1
t2 = 1
t3 = -1
t4 = 0
t5 = -1
targets = np.array([t1,t2,t3,t4,t5])

# data calculated in Ex1- a)
design_matrix = np.array([
                            [1, 1, 1, 1],
                            [1, 0, 0, 0],
                            [1, -2, 4, -8],
                            [1, 5, 25, 125],
                            [1, -5, 25, -125]
                        ])

calculate_regression_weights(design_matrix, targets)