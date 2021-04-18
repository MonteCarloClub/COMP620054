import numpy as np


def parse_iris_dat_to_mat():
    dat = np.loadtxt(fname='iris.dat', delimiter='\t')
    dat_mat = np.mat(dat)
    input_mat = np.array(dat_mat[:, 0:-1])
    output_key_mat = np.array(dat_mat[:, -1])
    return input_mat, output_key_mat
