import math
import pandas as pd

BIGINT = 1e3
EPSILON = 1e-15


def approx_equal_df(df1, df2):
    if df1.shape[0] != df2.shape[0] or df1.shape[1] != df2.shape[1]:
        return False
    for i in range(df1.shape[0]):
        for j in range(df1.shape[1]):
            if abs(df1.at[i, j]-df2.at[i, j]) > EPSILON:
                return False
    return True


def distance_of_column_vector_df(v1, v2):
    if v1.shape[0] != v2.shape[0] or v1.shape[1] != 1 or v2.shape[1] != 1:
        return -1
    distance = 0.0
    for i in range(v1.shape[0]):
        distance += (v1.at[i, 0]-v2.at[i, 0])**2
    distance = math.sqrt(distance)
    return distance


def match_count(output_v, output_key_v, rank_table):
    if output_v.shape[0] != output_key_v.shape[0] or output_v.shape[1] != 1 or output_key_v.shape[1] != 1:
        return 0
    key_v = pd.DataFrame(index=range(output_key_v.shape[0]), columns=range(1))
    for i in range(output_key_v.shape[0]):
        key_v.at[rank_table[i], 0] = output_key_v.at[i, 0]
    count = 0

    current_count = _match_count_ternary_column_vector_df(
        _convert_ternary_column_vector_df(output_v, 0, 1, 2), key_v)
    if current_count > count:
        count = current_count

    current_count = _match_count_ternary_column_vector_df(
        _convert_ternary_column_vector_df(output_v, 0, 2, 1), key_v)
    if current_count > count:
        count = current_count

    current_count = _match_count_ternary_column_vector_df(
        _convert_ternary_column_vector_df(output_v, 1, 0, 2), key_v)
    if current_count > count:
        count = current_count

    current_count = _match_count_ternary_column_vector_df(
        _convert_ternary_column_vector_df(output_v, 1, 2, 0), key_v)
    if current_count > count:
        count = current_count

    current_count = _match_count_ternary_column_vector_df(
        _convert_ternary_column_vector_df(output_v, 2, 0, 1), key_v)
    if current_count > count:
        count = current_count

    current_count = _match_count_ternary_column_vector_df(
        _convert_ternary_column_vector_df(output_v, 2, 1, 0), key_v)
    if current_count > count:
        count = current_count

    return count


def _match_count_ternary_column_vector_df(v1, v2):
    count = 0
    for i in range(v1.shape[0]):
        if v1.at[i, 0] == v2.at[i, 0]:
            count += 1
    return count


def _convert_ternary_column_vector_df(tcv_df, x1, x2, x3):
    converted_tcv_df = pd.DataFrame(
        index=range(tcv_df.shape[0]), columns=range(1))
    for i in range(tcv_df.shape[0]):
        if tcv_df.at[i, 0] == 0:
            converted_tcv_df.at[i, 0] = x1
            continue
        if tcv_df.at[i, 0] == 1:
            converted_tcv_df.at[i, 0] = x2
            continue
        converted_tcv_df.at[i, 0] = x3
    return converted_tcv_df
