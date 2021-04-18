import pandas as pd

from data_parser_df import parse_iris_dat_to_df
from data_parser_mat import parse_iris_dat_to_mat
from df_shuffler import shuffle_df
from fcm_clusterer import fcm
from math_utils import match_count
from spectral_clusterer import spectral_cluster

input_df, output_key_df = parse_iris_dat_to_df()
shuffled_df, rank_table = shuffle_df(input_df)
print('###### The shuffled input data frame is as follows: ######')
print(shuffled_df)

print('\n##############################')
print('###### Fuzzy Clustering ######')
print('##############################')

first_center, iter_count, center, output = fcm(shuffled_df, 3, 2)
count_fc = match_count(output, output_key_df, rank_table)
print('\n###### The randomly selected center points of the clusters are as follows: ######')
print(first_center)
print(
    f'\n###### After {iter_count} iteration(s), the clustering results are as follows: ######')
print(output)
print('\n###### The center points of the clusters are as follows: ######')
print(center)
print('\n###### The clustering accuracy of fuzzy clustering is: ######')
print(f'{count_fc}/{input_df.shape[0]} == {count_fc/input_df.shape[0]}\n')

print('#################################')
print('###### Spectral Clustering ######')
print('#################################')

input_mat, output_key_mat = parse_iris_dat_to_mat()
output_mat = spectral_cluster(input_mat)

output_mat_df = pd.DataFrame(index=range(150), columns=range(1))
output_key_mat_df = pd.DataFrame(index=range(150), columns=range(1))
rank_table = list(range(150))

for i in range(150):
    output_mat_df.at[i, 0] = output_mat[i]
    output_key_mat_df.at[i, 0] = output_key_mat[i][0]

count_sc = match_count(output_mat_df, output_key_mat_df, rank_table)
print('\n###### The result of spectral clustering is as follows: ######')
print(output_mat_df)
print('\n###### The clustering accuracy of spectral clustering is: ######')
print(f'{count_sc}/150 == {count_sc/150}')
