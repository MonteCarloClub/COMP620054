import pandas as pd

from data_parser import parse_iris_dat
from df_shuffler import shuffle_df
from fcm_clusterer import fcm
from math_utils import gen_adj_matrix_df
from math_utils import match_count
from spectral_clusterer import gen_knn_matrix
from spectral_clusterer import gen_normalized_laplacian_matrix
from spectral_clusterer import spectral_cluster

input_df, output_key_df = parse_iris_dat()
shuffled_df, rank_table = shuffle_df(input_df)
print('###### The shuffled input data frame is as follows: ######')
print(shuffled_df)

print('\n##############################')
print('###### Fuzzy Clustering ######')
print('##############################')

first_center, iter_count, center, output = fcm(shuffled_df, 3, 2)
count = match_count(output, output_key_df, rank_table)
print('\n###### The randomly selected center points of the clusters are as follows: ######')
print(first_center)
print(
    f'\n###### After {iter_count} iteration(s), the clustering results are as follows: ######')
print(output)
print('\n###### The center points of the clusters are as follows: ######')
print(center)
print('\n###### The clustering accuracy of fuzzy clustering is: ######')
print(f'{count}/{input_df.shape[0]} == {count/input_df.shape[0]}\n')

print('#################################')
print('###### Spectral Clustering ######')
print('#################################')

adj_matrix_df = gen_adj_matrix_df(shuffled_df)
print('###### The adjacency matrix is as follows: ######')
print(adj_matrix_df)

knn_matrix = gen_knn_matrix(adj_matrix_df.values)
print('\n###### The KNN adjacency matrix is as follows: ######')
print(knn_matrix)

normalized_laplacian_matrix = gen_normalized_laplacian_matrix(knn_matrix)
print('\n###### The normalized Laplacian matrix is as follows: ######')
print(normalized_laplacian_matrix)

sc_output = spectral_cluster(normalized_laplacian_matrix)
print('\n###### The sc_output are as follows: ######')
print(sc_output)

sc_output_df = pd.DataFrame(index=range(len(sc_output)), columns=range(1))
for i in range(len(sc_output)):
    sc_output_df.at[i, 0] = sc_output[i]
count = match_count(sc_output_df, output_key_df, rank_table)
print('\n###### The clustering accuracy of spectral clustering is: ######')
print(f'{count}/{input_df.shape[0]}')
