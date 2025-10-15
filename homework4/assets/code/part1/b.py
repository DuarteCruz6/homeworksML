import numpy as np

def calculate_distance(point, cluster):
    total = 0
    num = 0 
    for p in cluster["points"]:
        if not np.array_equal(p, point):
            total += np.linalg.norm(point - p)
            num+=1
    
    if not total: return 0
    
    return total/num



def calculate_silhouette(xs, clusters, debug):
    """
    x: [x1,x2,x3]
    clusters: [cluster1, cluster2]
    cluster1: {"mu": u1, "points": [x2,x3]}
    cluster2: {"mu": u2, "points": [x1]}
    """
    silhouettes = []
    for cluster in clusters:
        total_silhouette = 0
        total_points = len(cluster["points"])
        
        for point in cluster["points"]:
            numerator = calculate_distance(point, cluster)
            if debug: print("numerator",numerator)
            
            other_clusters = []
            for c in clusters:
                if c is not cluster:
                    other_clusters.append(c)
            
            denominator = 0
            for c in other_clusters:
                denominator += calculate_distance(point, c)
            if debug: print("denominator",denominator)
            
            point_silhouette = 1 - (numerator/denominator)
            if debug: print("point_silhouette",point_silhouette)
            
            total_silhouette+=point_silhouette
        
        if debug: print("cluster_silhouette",total_silhouette/total_points)
        silhouettes.append(total_silhouette/total_points)
    return silhouettes