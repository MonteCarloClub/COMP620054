import numpy as np


def get_dist(vector_1, vector_2):
    assert(len(vector_1) == len(vector_2))
    return np.sqrt(np.sum(np.square(vector_1 - vector_2)))
