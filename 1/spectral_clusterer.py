import numpy as np
from sklearn.cluster import KMeans


def gen_knn_matrix(adj_matrix, k=3, sigma=1.0):
    len_adj_matrix = len(adj_matrix)
    knn_matrix = np.zeros([len_adj_matrix, len_adj_matrix])
    for i in range(len_adj_matrix):
        dist_with_index = zip(adj_matrix[i], range(len_adj_matrix))
        dist_with_index = sorted(dist_with_index, key=lambda x: x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)]
        for j in neighbours_id:
            knn_matrix[i][j] = np.exp(-adj_matrix[i][j]/2/sigma/sigma)
            knn_matrix[j][i] = knn_matrix[i][j]
    return knn_matrix


def gen_normalized_laplacian_matrix(knn_matrix):
    degree_matrix = np.sum(knn_matrix, axis=1)
    laplacian_matrix = np.diag(degree_matrix)-knn_matrix
    sqrt_degree_matrix = np.diag(1.0/(degree_matrix**(0.5)))
    return np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)


def spectral_cluster(normalized_laplacian_matrix):
    eig_values, eig_vectors = np.linalg.eig(normalized_laplacian_matrix)
    eig_values = zip(eig_values, range(len(eig_values)))
    eig_values = sorted(eig_values, key=lambda x: x[0])
    sorted_eig_vectors = np.vstack(
        [eig_vectors[:, i] for (v, i) in eig_values[:]]).T
    # sorted_eig_vectors = np.vstack(
    #     [eig_vectors[:, i] for (v, i) in eig_values[:1000]]).T
    sorted_eig_vectors = np.asarray(sorted_eig_vectors).astype(float)
    sp_kmeans = KMeans(n_clusters=3).fit(sorted_eig_vectors)
    return sp_kmeans.labels_
