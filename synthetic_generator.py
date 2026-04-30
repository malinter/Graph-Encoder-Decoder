import torch
import networkx as nx
import random
from torch_geometric.data import Data

class SystemGraphGenerator:
    """
    Generates synthetic graphs representing common system architecture patterns/vulnerabilities.
    """
    def __init__(self, num_nodes=20):
        self.num_nodes = num_nodes

    def generate_broken_dependency(self):
        """Creates a chain with a missing link."""
        G = nx.path_graph(self.num_nodes)
        # Randomly remove an edge to break the chain
        edges = list(G.edges())
        if edges:
            G.remove_edge(*random.choice(edges))
        return self._to_pyg(G, label="broken_dependency")

    def generate_unauthorized_access(self):
        """Creates a segmented graph with an illegal bridge."""
        G = nx.complete_graph(self.num_nodes // 2) # Segment A (Public)
        G2 = nx.complete_graph(self.num_nodes // 2) # Segment B (Restricted)
        G = nx.disjoint_union(G, G2)
        
        # Add a single unauthorized "bridge" edge
        u = random.randint(0, self.num_nodes // 2 - 1)
        v = random.randint(self.num_nodes // 2, self.num_nodes - 1)
        G.add_edge(u, v)
        return self._to_pyg(G, label="unauthorized_access")

    def generate_redundant_loop(self):
        """Creates a tree with extra redundant edges forming loops."""
        G = nx.balanced_tree(r=2, h=3) # ~15 nodes
        # Add some redundant edges
        nodes = list(G.nodes())
        for _ in range(3):
            u, v = random.sample(nodes, 2)
            G.add_edge(u, v)
        return self._to_pyg(G, label="redundant_loop")

    def _to_pyg(self, G, label):
        # Convert to PyG Data object
        edge_index = torch.tensor(list(G.edges())).t().contiguous()
        # Ensure it's undirected if needed or handle directedness
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Simple identity features for now
        x = torch.eye(len(G.nodes()))
        
        # If node count is less than target, pad features
        if x.size(0) < self.num_nodes:
            padding = torch.zeros((self.num_nodes - x.size(0), x.size(0)))
            x = torch.cat([x, padding], dim=0)
            x = torch.cat([x, torch.zeros((self.num_nodes, self.num_nodes - x.size(1)))], dim=1)
            
        return Data(x=x, edge_index=edge_index, y=label)

def generate_dataset(n_total=500):
    generator = SystemGraphGenerator()
    dataset = []
    for i in range(n_total):
        choice = i % 3
        if choice == 0:
            dataset.append(generator.generate_broken_dependency())
        elif choice == 1:
            dataset.append(generator.generate_unauthorized_access())
        else:
            dataset.append(generator.generate_redundant_loop())
    
    print(f"Generated {len(dataset)} synthetic graphs.")
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset(500)
    torch.save(dataset, 'synthetic_system_graphs.pt')
