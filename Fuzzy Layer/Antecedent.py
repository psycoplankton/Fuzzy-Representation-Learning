from FUZZY import FuzzyLayer
import numpy as np
import hparams
class Antecedant(FuzzyLayer):
    """
    Antecedant part of the fuzzy logic system, it gives us membership functions the node embeddings.
    """
    def __init__(self, X, centroid_array, cluster_indices_array):
        super().__init__(X, centroid_array, cluster_indices_array)


    def dot_product(self):                                          #claculates the dot product of the embeddings and the centroids
        dot_products = np.zeros(size = (self.X.shape[0], hparams.k))
        row =0
        for node_embed in self.X:
            for i in range(self.centroid_array.shape[0]):
                y = np.matmul(node_embed, self.centroid_array[i])
            dot_products[row][i] += y
            row +=1
        return dot_products                                         #returns the array of dot products of each embedding with each of the k-centroids.
                                                                        #size = (number_of_nodes, number_of_cluster)

    def get_points(self, cluster_num, embed_axis):
        shape = self.cluster_indices_array[cluster_num].shape[0]                                                          #we specify which cluster's which embedding axis is required.
        points = np.zeros(shape = shape)                                #stores the points corresponding to the cluster_num and embed_axis.
        cluster_ids = self.cluster_indices_array[cluster_num]
        for id in range(shape):
            points[id] += self.X[cluster_ids[id]][embed_axis]
        return points

    def get_mean(self, cluster_num, embed_axis):
        return self.centroid_array[cluster_num][embed_axis]

    def get_variance(self, cluster_num, embed_axis):
        points = self.get_points(cluster_num = cluster_num,
                                 embed_axis = embed_axis)
        y=0
        mean = self.get_mean(cluster_num = cluster_num,
                             embed_axis = embed_axis)
        shape = 0
        for x in tuple(points.shape):
            shape = x
        for i in range(shape):
            y +=  (mean - points[i])**2
            y = y/(points.shape[0])
        variance = np.sqrt(y)
        return variance

    def get_variance_tensor(self, embed_axis):
        variance_tensor = np.zeros(shape = hparams.k)
        for i in range(hparams.k):
            variance = self.get_variance(cluster_num = i,
                                         embed_axis = embed_axis)
            variance_tensor[i] += variance
        #variance_tensor = variance_tensor/np.linalg.norm(variance_tensor)
        return variance_tensor

    def gaussianMF(self, cluster_num, embed_axis, element): #element is basically the value of embeddings at each dimension in each of the node_embeddings
        mean = self.get_mean(cluster_num = cluster_num,
                             embed_axis = embed_axis)
        variance_tensor = self.get_variance_tensor(embed_axis = embed_axis)
        if(variance_tensor[cluster_num] != 0):
            gaussian = (element - mean)**2/(2*((variance_tensor[cluster_num])**2))
        else:
            return 0
        return np.exp(-gaussian)

    def get_membership_array(self, embed_axis):
        membership_array = np.zeros(shape = (hparams.nnode_embed_dim, self.X.shape[0], hparams.k)) #(50, 5242, 5)
        for node_embed_num in range(self.X.shape[0]):
            for i in range(hparams.k):
                membership = self.gaussianMF(cluster_num = i,
                                             embed_axis = embed_axis,
                                             element = self.X[node_embed_num][embed_axis])
                membership_array[embed_axis][node_embed_num][i] += membership
        return membership_array[embed_axis]
    
    



