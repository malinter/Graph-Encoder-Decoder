import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

class GraphEncoder(nn.Module):
    """
    Encoder module that uses GCN or GAT to 
    generate latent representations of graph nodes.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout=0.2, use_gat=False):
        super(GraphEncoder, self).__init__()
        if use_gat:
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, latent_dim, heads=1, concat=False)
        else:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # Layer 1: GCN + Activation + Dropout
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2: GCN to Latent Space
        node_embeddings = self.conv2(x, edge_index)
        
        # If batch is provided, compute graph-level embedding
        if batch is not None:
            graph_embedding = global_mean_pool(node_embeddings, batch)
            return node_embeddings, graph_embedding
            
        return node_embeddings

class GraphDecoder(nn.Module):
    """
    Decoder module that attempts to reconstruct the graph structure
    (adjacency matrix) from node embeddings.
    """
    def __init__(self, latent_dim, dropout=0.2):
        super(GraphDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, z, edge_index_query=None):
        """
        By default, computes inner product between node embeddings 
        to predict the probability of an edge.
        """
        # Simple Inner Product Decoder
        if edge_index_query is not None:
            # Predict only for specific edges
            u, v = edge_index_query
            return (z[u] * z[v]).sum(dim=-1)
        
        # Compute full adjacency reconstruction (expensive for large graphs)
        adj_rec = torch.matmul(z, z.t())
        return torch.sigmoid(adj_rec)

class GEDCore(nn.Module):
    """
    Main Graph Encoder-Decoder (GED) class.
    """
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32, dropout=0.2, use_gat=False):
        super(GEDCore, self).__init__()
        self.encoder = GraphEncoder(input_dim, hidden_dim, latent_dim, dropout, use_gat)
        self.decoder = GraphDecoder(latent_dim, dropout)

    def forward(self, x, edge_index, batch=None):
        # Encode graph to latent space
        return self.encoder(x, edge_index, batch)

    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model weights saved to {path}")

    def load_weights(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model weights loaded from {path}")

    def reconstruct(self, z):
        # Reconstruct the graph from embeddings
        return self.decoder(z)

class GraphDataUtils:
    """
    Utility class for handling graph data conversion.
    """
    @staticmethod
    def to_pyg_data(nodes, edges):
        """
        Converts raw node features and edge lists into a torch_geometric.data.Data object.
        
        Args:
            nodes (Tensor): Node feature matrix [num_nodes, num_features]
            edges (Tensor): Edge index [2, num_edges]
        """
        from torch_geometric.data import Data
        return Data(x=nodes, edge_index=edges)
