import numpy as np
import math

def calculate_likelihood(x, cluster, debug):
    numerator = np.exp( -0.5 * np.transpose((x - cluster["mu"])) @ np.linalg.inv(cluster["sigma"]) @ (x - cluster["mu"]))
    if debug: print("numerator likelihood:", numerator)
    
    denominator = math.sqrt((2 * math.pi)**x.shape[0] * np.linalg.det(cluster["sigma"]))
    if debug: print("denominator likelihood:", denominator)
    
    return numerator / denominator
    
    
def e_step(xs, clusters, debug):
    
    posteriors = {}
    
    for i in range(len(xs)):
        if debug: print("X:",i+1)
        x = xs[i]
        
        probs = []
        
        for j in range(len(clusters)):
            if debug: print("clusters:",j+1)
            cluster = clusters[j]
            
            prior = cluster["p"]
            if debug: print("prior", prior)
            
            likelihood = calculate_likelihood(x, cluster, debug)
            if debug: print("likelihood", likelihood)
            
            joint_probability = prior*likelihood
            if debug: print("joint_probability", joint_probability)
            
            probs.append(joint_probability)
        
        total = sum(probs)
        for j in range(len(clusters)):
            pos = probs[j] / total
            if debug: print("cluster",j+1, "posterior:", pos)
            
            if posteriors.get(j):
                posteriors[j].append(pos)
            else:
                posteriors[j] = [pos]
            
    return posteriors

def calculate_mu(xs, posterior_list, debug):    
    numerator = 0
    denominator = 0
    
    for i in range(len(xs)):
        numerator+= posterior_list[i] * xs[i]
        denominator+= posterior_list[i]
    
    mu = numerator/denominator
    return mu

def calculate_covariance_matrix(xs, posterior_list, mu, debug):
    matrix = []
    
    for row in range(len(mu)):
        line = []
        
        for col in range(len(mu)):
            
            numerator = 0
            denominator = 0
            
            for i in range(len(xs)):
                posterior = posterior_list[i]
                x = xs[i]
                
                numerator += posterior * (x[row] - mu[row]) * (x[col] - mu[col])
                denominator += posterior
                
            line.append(numerator/denominator)   
            if debug: print(f"{row},{col}=",numerator/denominator)
        
        matrix.append(line)
        
    if debug: print("cov matrix", matrix)   
    return np.array(matrix)

        
def calculate_prior(posteriors, j):
    posterior_list = posteriors[j]
    
    numerator = sum(posterior_list)
    denominator = sum([sum(lst) for lst in posteriors.values()])
    
    return numerator/denominator
      
def m_step(xs, clusters, posteriors, debug):
    for j in range(len(clusters)):
        print("cluster:",j+1)
        
        cluster = clusters[j]
        posterior_list = posteriors[j]
        
        mu = calculate_mu(xs, posterior_list, debug)
        print("new mu", mu)
        
        cov_matrix = calculate_covariance_matrix(xs, posterior_list, mu, debug)
        print("new cov_matrix", cov_matrix)
        
        prior = calculate_prior(posteriors, j)
        print("new prior", prior)


def iterate(xs, clusters, debug):
    """
    x: [x1,x2,x3]
    clusters: [cluster1, cluster2]
    cluster1: {"p": p1, "mu": u1, "sigma": e1}
    cluster2: {"p": p2, "mu": u2, "sigma": e2}
    """
    posteriors = e_step(xs, clusters, debug)
    
    m_step(xs, clusters, posteriors, debug)
    



# data
x1 = np.transpose(np.array([0,0]))
x2 = np.transpose(np.array([2,0]))
x3 = np.transpose(np.array([0,3]))
xs = np.array([x1,x2,x3])

# cluster1 data
u1 = np.transpose(np.array([1,1]))
p1 = 0.6
e1 = np.array([[1,0],[0,1]])
cluster1 = {"p": p1, "mu": u1, "sigma": e1}

# cluster2 data
u2 = np.transpose(np.array([-1,-1]))
p2 = 0.4
e2 = np.array([[1,0],[0,1]])
cluster2 = {"p": p2, "mu": u2, "sigma": e2}

clusters = [cluster1, cluster2]
iterate(xs, clusters, debug = False)
    