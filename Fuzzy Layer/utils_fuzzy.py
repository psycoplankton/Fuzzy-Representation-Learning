import numpy as np
import os
import hparams
def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        rng = np.random.default_rng(seed = 42)
        embedding_matrix = rng.standard_normal(size = (n_node, n_embed))
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix

def str_list_to_float(str_list):
    return [float(item) for item in str_list]


