from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture


def k_means_cluster(X):
    """
    use kmeans method to cluster the data
    :param X:
    :return:
    """
    y_pred = KMeans(n_clusters=5, random_state=9).fit_predict(X)
    print(metrics.calinski_harabaz_score(X, y_pred))
    return y_pred


def dbscan_cluster(X):
    """
    use dbscan method to cluster the data
    :param X:
    :return:
    """
    y_pred = DBSCAN(eps=10, min_samples=10).fit_predict(X)
    print(metrics.calinski_harabaz_score(X, y_pred))
    return y_pred


def gmm_cluster(X):
    """
    use gmm method to cluster the data
    :param X:
    :return:
    """
    gmm = GaussianMixture(n_components=10)
    gmm.fit(X)
    y_pred = gmm.predict(X)
    print(metrics.calinski_harabaz_score(X, y_pred))
    return y_pred


def hierarchical_cluster(X):
    """
    use hierarchical method to cluster the data
    :param X:
    :return:
    """
    clu = AgglomerativeClustering(n_clusters=10,
                                  linkage='average')
    y_pred = clu.fit_predict(X)
    print(metrics.calinski_harabaz_score(X, y_pred))
    return y_pred
