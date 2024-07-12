from FUZZY import FuzzyLayer
import hparams
import torch
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
    
    



