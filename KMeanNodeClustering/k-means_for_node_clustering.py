import numpy as np

def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        embedding_matrix = np.random.rand(n_node, n_embed)
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix

def str_list_to_float(str_list):
    return [float(item) for item in str_list]

X = read_embeddings("G:\My Drive\AI and ML reading material\GraphGAN_Project\CA-GrQc_pre_train.emb",
                                     n_node = 5242,
                                     n_embed = 50)
print(X)

#K-Means Algorithm for node embeddings
#the embediddings are 1xd dimensional.
class KMeansNodeClustering:

  def __init__(self, k, node_embed_dim):
      self.k = k
      self.centroids = None
      self.node_embed_dim = node_embed_dim

  @staticmethod
  def euclidean_measure(node_embed, centroids):
      return np.sqrt(np.sum((centroids - node_embed)**2, axis = 0))

  def fit(self, X, node_embed_dim, max_iterations = 200):
      rng = np.random.default_rng()
      self.centroids = rng.standard_normal(self.k, self.node_embed_dim)

      for _ in range(max_iterations):
          y = []

          for node_embed in X:
                distance = KMeansNodeClustering.euclidean_measure(node_embed, self.centroids)
                cluster_num = np.argmin(distance)
                y.append(cluster_num)
            
          y = np.array(y)

          cluster_indices = []

          for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

          cluster_centres = []  

          for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centres.append(self.centroids[i])
                else:
                    cluster_centres.append(np.mean(X[indices], axis = 0)[0])

          if np.max(self.centroids - np.array(cluster_centres)) < 0.0001:
                break
          else:
                self.centroids = np.array((cluster_centres))   
      return y   
        

for i in range(50):
    centre = []
    y = 0
    for id, index_id in enumerate(indices):
        y += (X[index_id][i])
    mean = y/len(y)
    centre.append(mean)
cluster_centres.append(centre)
