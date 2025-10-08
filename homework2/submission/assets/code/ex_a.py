import numpy as np
from scipy.stats import norm

# data provided in the question
x1 = np.array([-2, -1, 1, 2, -1, 2, 3, 4])
x2 = np.array([1, 2, 3, 4, 0.4, 1.1, 1.5, 1.8])
classes = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
query = np.array([1, 2])

# compute class parameters automatically
unique_classes = np.unique(classes)
priors = {}
means = {}
stds = {}

# compute values for normal Gaussian probability density function
for cls in unique_classes:
    mask = (classes == cls)
    data = np.column_stack((x1[mask], x2[mask]))
    means[cls] = data.mean(axis=0)
    stds[cls] = data.std(axis=0, ddof=0)
    priors[cls] = len(data) / len(classes)

# compute normal Gaussian probability density function (PDF)
def class_likelihood(query, mean, std):
    return np.prod(norm.pdf(query, loc=mean, scale=std))

posteriors = {
    cls: class_likelihood(query, means[cls], stds[cls]) * priors[cls]
    for cls in unique_classes
}

print("Posterior values:", posteriors)
print("Predicted class:", max(posteriors, key=posteriors.get))
