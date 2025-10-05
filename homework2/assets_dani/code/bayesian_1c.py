import numpy as np

# --- GIVEN DATA ---
# Data for the categorical feature x3, separated by class
x3_A = np.array([0, 1, 1, 0]) # Values for Class A
x3_B = np.array([1, 1, 0, 1]) # Values for Class B

# Query value (x3 = True)
query_x3 = 1

print("--- Naive Bayes for Categorical Feature ---")
print(f"Query: x3 = {query_x3}\n")

# --- STEP 1: CALCULATE PRIOR PROBABILITIES ---
n_A = len(x3_A)
n_B = len(x3_B)
n_total = n_A + n_B

prior_A = n_A / n_total
prior_B = n_B / n_total

print("--- 1. Priors ---")
print(f"P(Class A) = {n_A}/{n_total} = {prior_A}")
print(f"P(Class B) = {n_B}/{n_total} = {prior_B}\n")


# --- STEP 2: CALCULATE LIKELIHOODS P(x3=1|C) ---
# This is done by counting the frequency of the query value in each class's data.

# For Class A
count_match_A = np.sum(x3_A == query_x3)
likelihood_A = count_match_A / n_A

# For Class B
count_match_B = np.sum(x3_B == query_x3)
likelihood_B = count_match_B / n_B

print("--- 2. Likelihoods ---")
print("For Class A:")
print(f"  Count(x3=1) = {count_match_A} out of {n_A} samples")
print(f"  P(x3=1|A) = {count_match_A}/{n_A} = {likelihood_A}\n")

print("For Class B:")
print(f"  Count(x3=1) = {count_match_B} out of {n_B} samples")
print(f"  P(x3=1|B) = {count_match_B}/{n_B} = {likelihood_B:.2f}\n")


# --- STEP 3: CALCULATE POSTERIOR PROBABILITY SCORES ---
posterior_A = likelihood_A * prior_A
posterior_B = likelihood_B * prior_B

print("--- 3. Posterior Scores ---")
print(f"Score(A) = P(x3=1|A) * P(A) = {likelihood_A} * {prior_A} = {posterior_A}")
print(f"Score(B) = P(x3=1|B) * P(B) = {likelihood_B:.2f} * {prior_B} = {posterior_B}\n")


# --- STEP 4: MAKE THE PREDICTION ---
predicted_class = 'A' if posterior_A > posterior_B else 'B'

print("--- 4. Conclusion ✅ ---")
print(f"Since Score(B) > Score(A) ({posterior_B} > {posterior_A}), the model predicts Class '{predicted_class}'.")