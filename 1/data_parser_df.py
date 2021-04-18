import pandas as pd


def parse_iris_dat_to_df():
    iris_dat = pd.read_csv('iris.dat', sep='\t', header=None)
    input_df = pd.DataFrame(index=range(150), columns=range(4))
    output_key_df = pd.DataFrame(index=range(150), columns=range(1))
    for i in range(150):
        input_df.at[i, 0] = iris_dat.at[i, 0]
        input_df.at[i, 1] = iris_dat.at[i, 1]
        input_df.at[i, 2] = iris_dat.at[i, 2]
        input_df.at[i, 3] = iris_dat.at[i, 3]
        output_key_df.at[i, 0] = iris_dat.at[i, 4]
    return input_df, output_key_df
