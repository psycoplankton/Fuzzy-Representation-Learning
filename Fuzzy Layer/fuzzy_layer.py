import pandas as pd
import networkx as nx
import torch

print(torch.__version__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()


dataset = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\bio-grid-human\bio-grid-human_train.txt"
df = pd.read_csv(dataset,
                sep = '\t',
                names = ["NodeIDfrom", "NodeIDto"],
                )
#create the graph networkx object from the above dataframe

G = nx.from_pandas_edgelist(df = df,
                             source = "NodeIDfrom",
                             target = "NodeIDto",
                             create_using=nx.Graph())

import numpy as np

def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    with open(filename, "r") as f:
        lines = f.readlines()[1:]  # skip the first line
        rng = np.random.default_rng(seed=42)
        embedding_matrix = rng.standard_normal(size = (n_node, n_embed))
        for line in lines:
            emd = line.split()
            embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
        return embedding_matrix

def str_list_to_float(str_list):
    return [float(item) for item in str_list]
filename = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\test_embeddings.emb"
X = read_embeddings(filename=filename,
                                     n_node = 9436,
                                     n_embed = 50)

rng = np.random.default_rng(seed = 42)

k=5
X = np.array(X)
node_embed_dim = 50
centroid = rng.standard_normal(size = (k, node_embed_dim))


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

  def dimensional_mean(cluster_num, node_embed_dim, cluster_indices, cluster_centres): #calculates the mean of of the arrays, dimension-wise
        axis_centre = np.zeros(node_embed_dim)
        for i in range(node_embed_dim):
            y = 0
            tup = cluster_indices[cluster_num].shape
            shape = tup[0]
            for x in range(shape):
                y += X[cluster_indices[cluster_num][x]][0][i]
            mean = np.mean(y)
            axis_centre[i] += mean
        axis_centre = axis_centre/np.linalg.norm(axis_centre)
        return cluster_centres.append(axis_centre)

  def fit(self, max_iterations = 200):
      rng = np.random.default_rng(seed = 69)
      self.centroid = (rng.standard_normal(size = (self.k, self.node_embed_dim)))
      self.centroid  = (self.centroid)/(np.max(self.centroid))
      for _ in range(max_iterations):
          y = []

          for node_embed in self.X:
                distance = KMeansNodeClustering.euclidean_measure(node_embed = node_embed,
                                                                  centroid = centroid)
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
                                                          node_embed_dim=node_embed_dim,
                                                          cluster_indices = cluster_indices,
                                                          cluster_centres=cluster_centres)

          self.centroid = np.array(cluster_centres)
      return y, cluster_indices
  
clustered_nodes = KMeansNodeClustering(X = X,
                                       k = k,
                                       node_embed_dim = node_embed_dim)
#cluster_indices_id stores the number_id of the clusters each embeddings is related to
#cluster_indices stores all the nodes that belong to one particular cluster, for all the clusters.
cluster_indices_id, cluster_indices = clustered_nodes.fit()

centroid_array = clustered_nodes.centroid #contains the centroids of all the clusteres.

cluster_indices = np.array(cluster_indices, dtype = object)
cluster_indices_array = np.array([np.squeeze(cluster_indices[0], axis = 1),
                                  np.squeeze(cluster_indices[1], axis = 1,),
                                  np.squeeze(cluster_indices[2], axis = 1),
                                  np.squeeze(cluster_indices[3], axis = 1),
                                  np.squeeze(cluster_indices[4], axis = 1)], dtype = object)

class FuzzyLayer:
    """
    This is the parent class which will contain variables for for Antecedant, Inference and Consequent classes.

    Args:
        X: The dataset which contains all the embeddings. shape(no_of_nodes, node_embed_dim)
        centroid_array: contains all the centroids of all the clusters. shape(no_of_clusters, node_embed_dim)
        cluster_indices_array: contains all the indices that belong to one particular cluster. shape(no_of_clusters, *number_of_indices_per_cluster*)
                                                                                               number_of_indices_per_cluster varies therefore it has no particular shape
    """

    def __init__(self, X, centroid_array, cluster_indices_array):
        self.X = X
        self.centroid_array = centroid_array
        self.cluster_indices_array = cluster_indices_array

import torch

class FuzzyLayer:
    """
    This is the parent class which will contain variables for Antecedant, Inference and Consequent classes.

    Args:
        X: The dataset which contains all the embeddings. shape(no_of_nodes, node_embed_dim)
        centroid_array: contains all the centroids of all the clusters. shape(no_of_clusters, node_embed_dim)
        cluster_indices_array: contains all the indices that belong to one particular cluster. shape(no_of_clusters, *number_of_indices_per_cluster*)
                                                                                               number_of_indices_per_cluster varies therefore it has no particular shape
    """

    def __init__(self, X, centroid_array, cluster_indices_array):
        self.X = torch.tensor(X, device=device, dtype=torch.float64)
        self.centroid_array = torch.tensor(centroid_array, device=device, dtype=torch.float64)
        self.cluster_indices_array = cluster_indices_array

class Antecedant(FuzzyLayer):
    """
    Antecedant part of the fuzzy logic system, it gives us membership functions for the node embeddings.
    """
    def __init__(self, X, centroid_array, cluster_indices_array):
        super().__init__(X, centroid_array, cluster_indices_array)

    def dot_product(self):
        # Use matrix multiplication instead of nested loops for efficiency
        dot_products = torch.matmul(self.X, self.centroid_array.T)
        return dot_products

    def get_points(self, cluster_num, embed_axis):
        # Fetch points corresponding to the cluster_num and embed_axis
        cluster_ids = torch.tensor(self.cluster_indices_array[cluster_num], device=device)
        points = self.X[cluster_ids, embed_axis]
        return points

    def get_mean(self, cluster_num, embed_axis):
        # Directly access the mean from centroid array
        return self.centroid_array[cluster_num, embed_axis]

    def get_standard_deviation(self, cluster_num, embed_axis):
        # Compute the variance in a vectorized manner
        points = self.get_points(cluster_num, embed_axis)
        mean = self.get_mean(cluster_num, embed_axis)
        variance = torch.var(points)
        variance = torch.sqrt(variance)
        return variance

    def get_stddev_tensor(self, embed_axis):
        # Vectorize variance computation for all clusters
        stddev = torch.tensor([self.get_standard_deviation(cluster_num, embed_axis) for cluster_num in range(self.centroid_array.shape[0])], device=device)
        return stddev

    def gaussianMF(self, cluster_num, embed_axis, element):
        # Vectorized Gaussian Membership Function calculation
        mean = self.get_mean(cluster_num, embed_axis)
        stddev_tensor = self.get_stddev_tensor(embed_axis)
        if stddev_tensor[cluster_num] != 0:
            gaussian = (element - mean) ** 2 / (2 * (stddev_tensor[cluster_num] ** 2))
        else:
            return 0.0
        return torch.exp(-gaussian)

    def get_membership_array(self, embed_axis):
        # Preallocate the membership array
        membership_array = torch.zeros((self.X.shape[0], self.centroid_array.shape[0]), device=device)
        for node_embed_num in range(self.X.shape[0]):
            for i in range(self.centroid_array.shape[0]):
                membership = self.gaussianMF(cluster_num=i, embed_axis=embed_axis, element=self.X[node_embed_num, embed_axis])
                membership_array[node_embed_num, i] = membership
        return membership_array


    
b = Antecedant(X = X,
               centroid_array = centroid_array,
               cluster_indices_array = cluster_indices_array)

membership_array = []
for embed_axis in range(node_embed_dim):
    mem_array_per_axis = b.get_membership_array(embed_axis = embed_axis)
    membership_array.append(mem_array_per_axis)
membership_array = np.array(membership_array, dtype = object)

class Consequent(FuzzyLayer):
    """
    This class generates the crisp embeddings using the membership values generated by the Antecedant class.

    Essentially, we fuzzified each of the embeddings for each node, clustered the nodes, found centroids,
    and calculated membership for each embedding. Now, we use that membership to output crisp memberships.

    To do that, we generate sets with variances and treat them as the consequent fuzzy set from which crisp output is generated.
    """

    def __init__(self, X, centroid_array, cluster_indices_array):
        super().__init__(X, centroid_array, cluster_indices_array)

    def get_average_membership_array(self, membership_array):
        """
        Calculates the average membership for each embedding dimension across all clusters.

        Args:
            membership_array (torch.Tensor): The membership values of shape (node_embed_dim, num_nodes, num_clusters).

        Returns:
            torch.Tensor: The average membership values of shape (node_embed_dim, num_nodes).
        """
        return torch.mean(membership_array, dim=2)  # shape = (node_embed_dim, num_nodes)

    def get_embed_axis_mean(self):
        """
        Calculates the mean of each embedding axis.

        Returns:
            torch.Tensor: The mean values for each embedding axis of shape (node_embed_dim,).
        """
        a = self.X.transpose(dim0=1, dim1=0) #shape = (50, 5242)
        return torch.mean(a, axis = 1) #shape = (50,)

    def get_embed_axis_variance(self):
        """
        Calculates the variance of each embedding axis.

        Returns:
            torch.Tensor: The variance values for each embedding axis of shape (node_embed_dim,).
        """
        a = self.X.transpose(dim0=1, dim1=0)
        mean = self.get_embed_axis_mean()
        variance = torch.zeros(size = (node_embed_dim,), device = device)
        for i in range(node_embed_dim):
            x = torch.sqrt(torch.mean((a[i] - mean[i])**2))
            variance[i] += x
        return variance #shape = (50,)

    def get_crisp_embeddings(self, average_membership_array):
        """
        Generates the crisp embeddings using the average membership values.

        Args:
            average_membership_array (torch.Tensor): The average membership values of shape (node_embed_dim, num_nodes).

        Returns:
            torch.Tensor: The crisp embeddings of shape (num_nodes, node_embed_dim).
        """
        a = self.X.transpose(dim0=1, dim1=0)
        mean = self.get_embed_axis_mean()
        variance = self.get_embed_axis_variance()
        new_X = torch.zeros(size = (node_embed_dim, self.X.shape[0]), device = device)
        for i in range(node_embed_dim):
            for j in range(self.X.shape[0]):
                if average_membership_array[i][j] == 0:
                    new_X[i][j] = a[i][j]
                else:
                    b = -1*torch.log(average_membership_array[i][j])
                    if(a[i][j] < 0):
                        b = -1*torch.sqrt(b)
                    else:
                        b = torch.sqrt(b)
                    b = b*variance[i] + mean[i]
                    new_X[i][j] = b
        new_X = torch.transpose(new_X, dim0=1, dim1=0)
        new_X.shape
        #new_X = new_X/np.linalg.norm(new_X, axis = 0)
        return new_X
    
c = Consequent(X = X,
               centroid_array=centroid_array,
               cluster_indices_array = cluster_indices_array)

average_membership = c.get_average_membership_array(membership_array = membership_array)

crisp = c.get_crisp_embeddings(average_membership_array = average_membership)
norm = np.linalg.norm(crisp, axis = 1)
print(norm.shape)
for i in range(crisp.shape[0]):
    crisp[i] = crisp[i]
crisp

import os
embedding_filename = r"C:\Users\vansh\AI and ML reading material\GraphGAN_Project\GraphGAN\Results\new.txt"
embeddings = crisp
index = np.array(range(5242)).reshape(-1, 1)
embedding_matrix = np.hstack([index, embeddings])
embedding_list = embedding_matrix.tolist()
embedding_str = [str(int(emb[0])) + " " + " ".join([str(x) for x in emb[1:]]) + "\n"
                  for emb in embedding_list]
with open(embedding_filename, "w+") as f:
    lines = [str(X.shape[0]) + "\t" + str(node_embed_dim) + "\n"] + embedding_str
    f.writelines(lines)



