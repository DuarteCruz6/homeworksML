import numpy as np
from scipy.stats import multivariate_normal

query = np.array([1, 2])

mu_A = np.array([0, 2.5])
sigma_A = np.array([[10/3, 7/3],
                  [7/3, 5/3]])
pdf_value_A = multivariate_normal.pdf(query, mean=mu_A, cov=sigma_A)

mu_B = np.array([2, 1.2])
sigma_B = np.array([[14/3, 1.3],
                  [1.3, 11/30]])

pdf_value_B = multivariate_normal.pdf(query, mean=mu_B, cov=sigma_B)

print("Numerator for Class = A:", pdf_value_A)
print("Numerator for Class = B:", pdf_value_B)
