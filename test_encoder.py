import torch
from models.ged_core import GEDCore

def test_encoder():
    # Setup parameters
    input_dim = 16
    hidden_dim = 32
    latent_dim = 8
    num_nodes = 10
    
    # Initialize model
    model = GEDCore(input_dim, hidden_dim, latent_dim)
    model.eval()
    
    # Create dummy graph data
    # x: Node features [num_nodes, input_dim]
    x = torch.randn((num_nodes, input_dim))
    
    # edge_index: Graph connectivity [2, num_edges]
    # Creating a simple cycle graph
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    ], dtype=torch.long)
    
    print(f"Input features shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    
    # Forward pass
    with torch.no_grad():
        node_embeddings = model(x, edge_index)
        
    print(f"Node embeddings shape: {node_embeddings.shape}")
    
    if node_embeddings.shape == (num_nodes, latent_dim):
        print("Encoder test PASSED!")
    else:
        print("Encoder test FAILED: Unexpected output shape.")

if __name__ == "__main__":
    test_encoder()
