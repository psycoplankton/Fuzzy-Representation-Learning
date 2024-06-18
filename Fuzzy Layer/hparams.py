import torch

node_embed_dim = 50 # number of embedding dimensions
k = 3 # number of clusters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #device 
embedding_filename = r"G:\My Drive\AI and ML reading material\GraphGAN_Project\testing.emb"