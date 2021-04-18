import numpy as np
import pandas as pd
import random


def shuffle_df(df):
    rank_table = list(range(df.shape[0]))
    random.shuffle(rank_table)
    shuffled_df = pd.DataFrame(index=range(
        df.shape[0]), columns=range(df.shape[1]))
    for i in range(len(rank_table)):
        for j in range(df.shape[1]):
            shuffled_df.at[rank_table[i], j] = df.at[i, j]
    return shuffled_df, rank_table
