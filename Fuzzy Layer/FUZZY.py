import torch
import hparams
import hparams
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
        self.X = torch.tensor(X, device=hparams.device, dtype=torch.float64)
        self.centroid_array = torch.tensor(centroid_array, device=hparams.device, dtype=torch.float64)
        self.cluster_indices_array = cluster_indices_array