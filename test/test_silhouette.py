from sklearn.metrics import silhouette_score
import numpy as np
import cluster
import pytest

# write your silhouette score unit tests here

def test_compare_to_sklearn():

    """Check that mean of silhouette scores is close to sklearn calculation"""

    d_clusters, d_labels = cluster.make_clusters(n=1000, m=200, k=3)
    silhouette = cluster.Silhouette()
    scores = silhouette.score(d_clusters, d_labels)

    #compare to sklearn scores
    assert abs(silhouette_score(d_clusters, d_labels)-np.mean(scores)) < 0.001

def test_size_check():

    "Check that error is raised for dimension mismatch"

    X = np.random.rand(100,20)
    Y = np.random.rand(50)

    silhouette = cluster.Silhouette()

    with pytest.raises(Exception):
        assert silhouette.score(X, Y), "Exception should be raised for dimension mismatch"
