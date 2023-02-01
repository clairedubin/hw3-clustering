# Write your k-means unit tests here
import numpy as np
import cluster
import pytest

def test_function_order():

    """Check that error is thrown if you try to get error before predict is run"""

    d_clusters, d_labels = cluster.make_clusters(n=1000, m=200, k=3)
    kmeans = cluster.KMeans(k=3)

    with pytest.raises(Exception):
        assert kmeans.get_error(), "Error should be thrown if trying to access error before attribute is initialized"

def test_size_check():

    "Check that error is thrown if input matrix has fewer than k entries"

    d_clusters, d_labels = cluster.make_clusters(n=4, m=200, k=3)
    kmeans = cluster.KMeans(k=5)

    with pytest.raises(Exception):
        assert kmeans.predict(d_clusters), "Error should be thrown if matrix has fewer than k rows"


def test_num_clusters_output():

    "Check that the number of output clusters == k"
 
    k=3
    d_clusters, d_labels = cluster.make_clusters(n=1000, m=200, k=k)

    kmeans = cluster.KMeans(k=k)
    predicted_labels=kmeans.predict(d_clusters)

    assert len(np.unique(predicted_labels) == k), "Not the expected number of output clusters"