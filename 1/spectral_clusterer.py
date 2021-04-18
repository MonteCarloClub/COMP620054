from sklearn.cluster import SpectralClustering


def spectral_cluster(input_mat):
    output_mat = SpectralClustering(
        gamma=0.1, n_clusters=3).fit_predict(input_mat)
    return output_mat
