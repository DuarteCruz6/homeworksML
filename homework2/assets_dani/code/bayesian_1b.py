import numpy as np

# A function for the Gaussian Probability Density Function (PDF)
def gaussian_pdf(x, mean, var):
    """
    Calculates the probability of x for a given 1D Gaussian distribution.
    
    Args:
        x (float): The value to evaluate.
        mean (float): The mean of the Gaussian distribution.
        var (float): The variance of the Gaussian distribution.
        
    Returns:
        float: The probability density of x.
    """
    if var == 0:
        return 1 if x == mean else 0 # Handle zero variance case
    
    coefficient = 1.0 / np.sqrt(2 * np.pi * var)
    exponent = np.exp(-((x - mean)**2) / (2 * var))
    return coefficient * exponent

# --- GIVEN DATA ---
# Dataset for Class A and Class B
data_A = np.array([[-2, 1], [-1, 2], [1, 3], [2, 4]], dtype=float)
data_B = np.array([[-1, 0.4], [2, 1.1], [3, 1.5], [4, 1.8]], dtype=float)

# Query vector
query_x = np.array([1, 2], dtype=float)

print("--- Naive Bayes Classification ---")
print(f"Query Vector: x = {query_x}\n")

# --- STEP 1: CALCULATE PRIOR PROBABILITIES ---
n_A = len(data_A)
n_B = len(data_B)
n_total = n_A + n_B

prior_A = n_A / n_total
prior_B = n_B / n_total

print("--- 1. Priors ---")
print(f"P(Class A) = {n_A}/{n_total} = {prior_A}")
print(f"P(Class B) = {n_B}/{n_total} = {prior_B}\n")

# --- STEP 2: CALCULATE LIKELIHOOD PARAMETERS (MEAN & VARIANCE) ---
# Parameters for Class A
mean_A = np.mean(data_A, axis=0)
var_A = np.var(data_A, axis=0) # Use population variance (ddof=0 is default)

# Parameters for Class B
mean_B = np.mean(data_B, axis=0)
var_B = np.var(data_B, axis=0)

print("--- 2. Gaussian Parameters ---")
print(f"Class A: Mean = {mean_A}, Variance = {var_A}")
print(f"Class B: Mean = {mean_B}, Variance = {var_B}\n")

# --- STEP 3: CALCULATE LIKELIHOODS FOR THE QUERY VECTOR ---
# Likelihood for Class A
likelihood_x1_A = gaussian_pdf(query_x[0], mean_A[0], var_A[0])
likelihood_x2_A = gaussian_pdf(query_x[1], mean_A[1], var_A[1])
total_likelihood_A = likelihood_x1_A * likelihood_x2_A

# Likelihood for Class B
likelihood_x1_B = gaussian_pdf(query_x[0], mean_B[0], var_B[0])
likelihood_x2_B = gaussian_pdf(query_x[1], mean_B[1], var_B[1])
total_likelihood_B = likelihood_x1_B * likelihood_x2_B

print("--- 3. Likelihoods P(x|C) ---")
print("For Class A:")
print(f"  P(x1=1|A) = {likelihood_x1_A:.4f}")
print(f"  P(x2=2|A) = {likelihood_x2_A:.4f}")
print(f"  P(x|A) = {likelihood_x1_A:.4f} * {likelihood_x2_A:.4f} = {total_likelihood_A:.4f}\n")

print("For Class B:")
print(f"  P(x1=1|B) = {likelihood_x1_B:.4f}")
print(f"  P(x2=2|B) = {likelihood_x2_B:.4f}")
print(f"  P(x|B) = {likelihood_x1_B:.4f} * {likelihood_x2_B:.4f} = {total_likelihood_B:.4f}\n")

# --- STEP 4: CALCULATE POSTERIOR PROBABILITY SCORE ---
posterior_A = total_likelihood_A * prior_A
posterior_B = total_likelihood_B * prior_B

print("--- 4. Posterior Scores P(x|C)P(C) ---")
print(f"Score(A) = P(x|A) * P(A) = {total_likelihood_A:.4f} * {prior_A} = {posterior_A:.4f}")
print(f"Score(B) = P(x|B) * P(B) = {total_likelihood_B:.4f} * {prior_B} = {posterior_B:.4f}\n")

# --- STEP 5: MAKE THE PREDICTION ---
predicted_class = 'A' if posterior_A > posterior_B else 'B'

print("--- 5. Conclusion ---")
print(f"Since Score(A) > Score(B) ({posterior_A:.4f} > {posterior_B:.4f}), the model predicts Class '{predicted_class}'.")