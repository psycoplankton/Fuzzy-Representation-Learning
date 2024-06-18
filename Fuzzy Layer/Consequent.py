from FUZZY import FuzzyLayer
import numpy as np
import hparams

class Consequent(FuzzyLayer):
    """
    This class generates the the crisp embeddings using the membership that were generated using the ANtecedant class.

    Essentially, we were able to fuzzify each of the embeddings for each of the node. We clustered the nodes, found 5 centroids, found out
    membership for each of the embedding and now we look to use that membership to output crisp memberships.

    To do that we will generate 50 sets with 50 variances and we will treat it as the consequent fuzzy set from which crisp output is generated.
    """
    def __init__(self, X, centroid_array, cluster_indices_array):
        super().__init__(X, centroid_array, cluster_indices_array)

    def get_average_membership_array(self, membership_array):
        return np.mean(membership_array, axis = 2) #shape = (50, 5242)

    def get_embed_axis_mean(self):
        a = self.X.transpose() #shape = (50, 5242)
        return np.mean(a, axis = 1) #shape = (50,)

    def get_embed_axis_variance(self):
        a = self.X.transpose()
        mean = self.get_embed_axis_mean()
        variance = np.zeros(shape = hparams.node_embed_dim)
        for i in range(hparams.node_embed_dim):
            x = np.sqrt(np.mean((a[i] - mean[i])**2))
            variance[i] += x
        return variance #shape = (50,)

    def get_crisp_embeddings(self, average_membership_array): #average_membership_array.shape = (50, 5242)
        a = self.X.transpose()
        mean = self.get_embed_axis_mean()
        variance = self.get_embed_axis_variance()
        new_X = np.zeros(shape = (hparams.node_embed_dim, self.X.shape[0]))
        for i in range(hparams.node_embed_dim):
            for j in range(self.X.shape[0]):
                if average_membership_array[i][j] == 0:
                    new_X[i][j] = a[i][j]
                else:
                    b = -1*np.log(average_membership_array[i][j])
                    if(a[i][j] < 0):
                        b = -1*np.sqrt(b)
                    else:
                        b = np.sqrt(b)
                    b = b*variance[i] + mean[i]
                    new_X[i][j] = b
        new_X = new_X.transpose()
        new_X.shape
        #new_X = new_X/np.linalg.norm(new_X, axis = 0)
        return new_X



