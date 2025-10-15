from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score

def evaluate(X, em_model):
    labels_em= em_model.predict(X)
    print("Silhouette:",metrics.silhouette_score(X, labels_em, metric='euclidean'))
    print("Davies Bouldin:",davies_bouldin_score(X, labels_em))

def do_em(X, n_components, print_stuff):

    # learn EM with multivariate Gaussian assumption
    em_algo = GaussianMixture(n_components=n_components, covariance_type='full',n_init=1) 
    em_model = em_algo.fit(X)

    # describe EM solution
    if print_stuff: print("means:\n",em_model.means_,"\n\ncovariances:\n",em_model.covariances_)
    
    return em_model.predict_proba(X), em_model

def do_em_pca(X, pca, n_components):
    X_pca = pca.transform(X)
    
    em_algo = GaussianMixture(n_components=n_components, covariance_type='full',n_init=1) 
    
    em_model = em_algo.fit(X_pca)
    labels_em= em_model.predict(X_pca)
    
    return labels_em
