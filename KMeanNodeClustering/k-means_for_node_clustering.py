import numpy as np
import hparams

#K-Means Algorithm for node embeddings
#the embediddings are 1xd dimensional.
class KMeansNodeClustering:

  def __init__(self, X, k, node_embed_dim):
      self.k = k #k is the number of cluster centres
      self.X = X #dataset consisting of node embeddings
      self.node_embed_dim = node_embed_dim #dimensions of node embeddings.
      self.centroid = None #these will be the centres of our distribution

  @staticmethod
  def euclidean_measure(centroid, node_embed):  # calculates the distance of the k-dimensional node from the centre
      return np.sqrt(np.sum((centroid - node_embed)**2, axis=1))

  def dimensional_mean(self,cluster_num, node_embed_dim, cluster_indices, cluster_centres): #calculates the mean of of the arrays, dimension-wise
        axis_centre = np.zeros(node_embed_dim)
        for i in range(node_embed_dim):
            y = 0
            tup = cluster_indices[cluster_num].shape
            shape = tup[0]
            for x in range(shape):
                y += self.X[cluster_indices[cluster_num][x]][0][i]
            mean = np.mean(y)
            axis_centre[i] += mean
        axis_centre = axis_centre/np.linalg.norm(axis_centre)
        return cluster_centres.append(axis_centre)

  def fit(self, max_iterations = 200):
      rng = np.random.default_rng(seed = 69)
      self.centroid = rng.standard_normal(size = (self.k, self.node_embed_dim))
      self.centroid  = (self.centroid)/(np.max(self.centroid))
      for _ in range(max_iterations):
          y = []

          for node_embed in self.X:
                distance = KMeansNodeClustering.euclidean_measure(node_embed = np.array(node_embed),
                                                                  centroid = self.centroid)
                cluster_num = np.argmin(distance)
                y.append(cluster_num)

          y = np.array(y) #stores the clustur number each of the nodes belong to

          cluster_indices = [] #to know which node belongs to which cluster

          for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i)) #returns every node which belongs to the same cluster
          cluster_indices = np.array(cluster_indices, dtype = object)
          cluster_centres = [] #stores the centres of the clusters

          for j, indices in enumerate(cluster_indices): #cluster_indices contains the cluster numbers and the indices that belong to a particular cluster
                #i = cluster number
                #indices = indices of the nodes that belong to i.
                if len(indices) == 0:
                    cluster_centres.append(self.centroid[i])
                else:
                    KMeansNodeClustering.dimensional_mean(
                                                          cluster_num = j,
                                                          node_embed_dim=hparams.node_embed_dim,
                                                          cluster_indices = cluster_indices,
                                                          cluster_centres=cluster_centres)

          self.centroid = np.array(cluster_centres)
          y = np.array(y)
      return y, cluster_indices


