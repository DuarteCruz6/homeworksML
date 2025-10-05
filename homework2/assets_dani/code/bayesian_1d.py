# Step 1: Define the given values (Priors and Likelihoods)

# Priors
p_A = 0.5
p_B = 0.5

# Likelihoods for the dependable features (x1, x2) = (1, 2)
# p((x1, x2)|C) from section (b)
p_x12_given_A = 0.0667
p_x12_given_B = 0.0439

# Likelihoods for the independent categorical feature x3 = 1
# P(x3=1|C) from section (c)
p_x3_given_A = 0.5
p_x3_given_B = 0.75

print("--- Defined Values ---")
print(f"P(A) = {p_A}")
print(f"P(B) = {p_B}")
print(f"p(x1, x2|A) = {p_x12_given_A}")
print(f"p(x1, x2|B) = {p_x12_given_B}")
print(f"P(x3=1|A) = {p_x3_given_A}")
print(f"P(x3=1|B) = {p_x3_given_B}")
print("----------------------\n")

# Step 2: Compute the joint probability for Class A
# p(A, x_query) = p((x1,x2)|A) * P(x3=1|A) * P(A)
p_A_xquery = p_x12_given_A * p_x3_given_A * p_A

print("--- Joint Probability for Class A ---")
print(f"Calculation: {p_x12_given_A} * {p_x3_given_A} * {p_A}")
print(f"p(A, x_query) = {p_A_xquery}")
print("-------------------------------------\n")

# Step 3: Compute the joint probability for Class B
# p(B, x_query) = p((x1,x2)|B) * P(x3=1|B) * P(B)
p_B_xquery = p_x12_given_B * p_x3_given_B * p_B

print("--- Joint Probability for Class B ---")
print(f"Calculation: {p_x12_given_B} * {p_x3_given_B} * {p_B}")
print(f"p(B, x_query) = {p_B_xquery}")
print("-------------------------------------\n")

# Step 4: Determine the most probable class
most_probable_class = 'A' if p_A_xquery > p_B_xquery else 'B'

print(f"--- Comparison: p(A, x_query) vs p(B, x_query) ---")
print(f"Result: {p_A_xquery} > {p_B_xquery} is {p_A_xquery > p_B_xquery}")
print(f"The most probable class is: {most_probable_class}")
print("---------------------------------------------------\n")

# Step 5: Compute the estimated relative probability (Ratio of higher to lower probability)
if p_A_xquery > p_B_xquery:
    relative_probability = p_A_xquery / p_B_xquery
    ratio_label = "p(A, x_query) / p(B, x_query)"
else:
    relative_probability = p_B_xquery / p_A_xquery
    ratio_label = "p(B, x_query) / p(A, x_query)"

print("--- Estimated Relative Probability ---")
print(f"Ratio: {ratio_label}")
print(f"Relative probability = {relative_probability}")
print("--------------------------------------")