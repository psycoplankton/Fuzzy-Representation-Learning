import numpy as np
import os
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

def write_embeddings_to_file():
    embedding_filename = r"G:\My Drive\AI and ML reading material\GraphGAN_Project\testing.emb"
    embeddings = crisp
    index = np.array(range(5242)).reshape(-1, 1)
    embedding_matrix = np.hstack([index, embeddings])
    embedding_list = embedding_matrix.tolist()
    embedding_str = [str(int(emb[0])) + " " + " ".join([str(x) for x in emb[1:]]) + "\n"
                      for emb in embedding_list]
    with open(embedding_filename, "w+") as f:
        lines = [str(X.shape[0]) + "\t" + str(node_embed_dim) + "\n"] + embedding_str
        f.writelines(lines)
