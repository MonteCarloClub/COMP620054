import pandas as pd
import random

from math_utils import BIGINT
from math_utils import EPSILON
from math_utils import approx_equal_df
from math_utils import distance_of_column_vector_df


def fcm(df, cluster_count, membership_factor):
    weight_df = gen_weight_df(df.shape[0], cluster_count)

    iter_count = 0
    center = pd.DataFrame(index=range(cluster_count),
                          columns=range(df.shape[1]))
    distance_df = pd.DataFrame(index=range(
        df.shape[0]), columns=range(cluster_count))
    while(True):
        iter_count += 1

        old_weight_df = weight_df.copy()

        for i in range(cluster_count):
            for j in range(df.shape[1]):
                numrtr, denmntr = 0.0, 0.0
                for k in range(df.shape[0]):
                    numrtr += (weight_df.at[k, i] **
                               membership_factor)*df.at[k, j]
                    denmntr += (weight_df.at[k, i]**membership_factor)
                center.at[i, j] = numrtr/denmntr

        if iter_count == 1:
            first_center = center.copy()

        for i in range(df.shape[0]):
            for j in range(cluster_count):
                v1 = pd.DataFrame(index=range(df.shape[1]), columns=range(1))
                v2 = pd.DataFrame(index=range(df.shape[1]), columns=range(1))
                for k in range(df.shape[1]):
                    v1.at[k, 0] = df.at[i, k]
                    v2.at[k, 0] = center.at[j, k]
                distance_df.at[i, j] = distance_of_column_vector_df(v1, v2)

        for i in range(df.shape[0]):
            for j in range(cluster_count):
                denmntr_sum = 0.0
                for k in range(cluster_count):
                    if distance_df.at[i, k] < EPSILON:
                        distance_df.at[i, k] = EPSILON
                    denmntr_sum += (distance_df.at[i, j] /
                                    distance_df.at[i, k])**(2/(membership_factor-1))
                weight_df.at[i, j] = 1/denmntr_sum

        if approx_equal_df(old_weight_df, weight_df):
            break

    return first_center, iter_count, center, gen_output(weight_df)


def gen_weight_df(df_len, cluster_count):
    weight_df = pd.DataFrame(index=range(df_len), columns=range(cluster_count))
    for i in range(df_len):
        rand_sum = 0
        for j in range(cluster_count):
            rand_elem = random.randint(1, BIGINT)
            weight_df.at[i, j] = rand_elem
            rand_sum += rand_elem
        for j in range(cluster_count):
            weight_df.at[i, j] = weight_df.at[i, j]/rand_sum
    return weight_df


def gen_output(weight_df):
    output_df = pd.DataFrame(index=range(weight_df.shape[0]), columns=range(1))
    for i in range(weight_df.shape[0]):
        if weight_df.at[i, 0] > weight_df.at[i, 1] and weight_df.at[i, 0] > weight_df.at[i, 2]:
            output_df.at[i, 0] = 0
            continue
        if weight_df.at[i, 1] > weight_df.at[i, 0] and weight_df.at[i, 1] > weight_df.at[i, 2]:
            output_df.at[i, 0] = 1
            continue
        output_df.at[i, 0] = 2
    return output_df
