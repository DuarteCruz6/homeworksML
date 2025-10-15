from sklearn import datasets, metrics, cluster, mixture
from sklearn.metrics import davies_bouldin_score

def evaluate(X, labels):
    # compute silhouette
    print("Silhouette:",metrics.silhouette_score(X, labels, metric='euclidean'))
    print("Davies Bouldin:",davies_bouldin_score(X, labels))


def do_kmeans(X, n_clusters, print_stuff):
    # parameterize clustering
    kmeans_algo = cluster.KMeans(n_clusters=n_clusters,algorithm='lloyd',init='random',n_init=1)

    # learn the model
    kmeans_model = kmeans_algo.fit(X)

    # return centroids
    kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    
    if print_stuff: print("means:\n",kmeans_model.cluster_centers_)
    
    return labels

def do_kmeans_pca(X, pca, n_clusters, print_stuff):
    X_pca = pca.transform(X)

    # learn the model
    kmeans_algo = cluster.KMeans(n_clusters=n_clusters,algorithm='lloyd',init='random',n_init=1)
    kmeans_model = kmeans_algo.fit(X_pca)

    # return centroids
    kmeans_model.cluster_centers_
    labels = kmeans_model.labels_
    
    if print_stuff: print("means:\n",kmeans_model.cluster_centers_)
    
    return labels
