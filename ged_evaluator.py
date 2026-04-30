import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from models.ged_core import GEDCore
import os
import csv
from datetime import datetime
import numpy as np
import networkx as nx

def calculate_metrics(adj_true, adj_pred, threshold=0.5):
    """
    Calculates Accuracy, Recall, and F1 Score for graph reconstruction.
    """
    pred = (adj_pred > threshold).float()
    
    tp = (pred * adj_true).sum().item()
    fp = (pred * (1 - adj_true)).sum().item()
    fn = ((1 - pred) * adj_true).sum().item()
    tn = ((1 - pred) * (1 - adj_true)).sum().item()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    return accuracy, recall, f1

def export_graph_to_graphml(data, filename):
    """
    Saves a PyG Data object as a GraphML file using NetworkX.
    """
    G = nx.Graph()
    edge_index = data.edge_index.cpu().numpy()
    for i in range(edge_index.shape[1]):
        G.add_edge(int(edge_index[0, i]), int(edge_index[1, i]))
    
    # Add node features as attributes if desired
    # For simplicity, just adding node IDs
    for node in G.nodes():
        G.nodes[node]['id'] = str(node)

    os.makedirs('telemetry', exist_ok=True)
    path = os.path.join('telemetry', filename)
    nx.write_graphml(G, path)
    print(f"Graph exported to {path}")

def generate_mermaid_graph(data, num_nodes=15):
    """
    Generates a Mermaid.js graph definition from PyG data.
    """
    print("\n--- Mermaid.js Graph Definition ---")
    print("graph TD")
    
    # Define labels (mock names as requested: MacBook, Azure, VPN, etc.)
    # In a real system, these would come from metadata
    sys_names = ["MacBook", "Azure_VPN", "Prod_DB", "Auth_Service", "JumpBox", 
                 "Azure_Foundry", "Telemetry_Node", "Proxy", "Firewall", "Storage_Blob",
                 "User_Node", "Admin_Node", "REST_API", "Worker_Service", "Gateway"]
    
    edge_index = data.edge_index.cpu().numpy()
    edges_found = set()
    
    # Print edges
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        if u < num_nodes and v < num_nodes:
            # Avoid duplicate undirected edges in Mermaid
            edge = tuple(sorted((u, v)))
            if edge not in edges_found:
                u_name = sys_names[u] if u < len(sys_names) else f"Node_{u}"
                v_name = sys_names[v] if v < len(sys_names) else f"Node_{v}"
                print(f"    {u_name} --- {v_name}")
                edges_found.add(edge)
    print("------------------------------------\n")

def print_ascii_graph(data, num_nodes=10):
    """
    Prints a simple adjacency list representation for quick inspection.
    """
    print("\n--- ASCII Graph Representation (Sample) ---")
    edge_index = data.edge_index.cpu().numpy()
    adj_list = {i: [] for i in range(num_nodes)}
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i]
        if u < num_nodes and v < num_nodes:
            adj_list[u].append(v)
    
    for u, neighbors in adj_list.items():
        print(f"Node {u:02d} -> {sorted(list(set(neighbors)))}")
    print("-------------------------------------------\n")

def evaluate_model():
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{timestamp} Starting Evaluation...")

    # 1. Load Data and Model
    if not os.path.exists('synthetic_system_graphs.pt'):
        print("Error: synthetic_system_graphs.pt not found.")
        return
    
    dataset = torch.load('synthetic_system_graphs.pt', weights_only=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    input_dim = dataset[0].x.size(1)
    model = GEDCore(input_dim=input_dim, hidden_dim=128, latent_dim=32, use_gat=True)
    
    if os.path.exists('weights/ged_foundation.pth'):
        model.load_weights('weights/ged_foundation.pth')
    else:
        print("Warning: weights/ged_foundation.pth not found. Evaluating uninitialized model.")

    model.eval()
    
    # 2. Compute Metrics
    all_acc, all_rec, all_f1 = [], [], []
    
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            if isinstance(out, tuple):
                node_z, _ = out
            else:
                node_z = out
            
            adj_pred = model.reconstruct(node_z)
            
            num_nodes = data.x.size(0)
            adj_true = torch.zeros((num_nodes, num_nodes))
            adj_true[data.edge_index[0], data.edge_index[1]] = 1.0
            
            acc, rec, f1 = calculate_metrics(adj_true, adj_pred)
            all_acc.append(acc)
            all_rec.append(rec)
            all_f1.append(f1)
            
    avg_acc = np.mean(all_acc)
    avg_rec = np.mean(all_rec)
    avg_f1 = np.mean(all_f1)
    
    print(f"Evaluation Results:")
    print(f"  Accuracy: {avg_acc:.4f}")
    print(f"  Recall:   {avg_rec:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    
    # 3. Visual/Structural Export
    sample_data = dataset[0]
    print_ascii_graph(sample_data)
    generate_mermaid_graph(sample_data)
    # export_graph_to_graphml(sample_data, "sample_graph.graphml") # Disabled as per request

    # 4. Plotting (Requires Matplotlib)
    try:
        import matplotlib.pyplot as plt
        
        # Load logs for plotting
        epochs, losses = [], []
        if os.path.exists('telemetry/training_log.csv'):
            with open('telemetry/training_log.csv', mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    epochs.append(int(row['Epoch']))
                    losses.append(float(row['Loss']))
        
        if epochs:
            plt.figure(figsize=(12, 5))
            
            # Plot 1: Loss vs Epochs
            plt.subplot(1, 2, 1)
            plt.plot(epochs, losses, color='blue', label='Training Loss')
            plt.title('Loss vs Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot 2: Metrics (Mockup if historical metrics not logged, 
            # or just show final metrics as points)
            plt.subplot(1, 2, 2)
            plt.bar(['Accuracy', 'Recall', 'F1'], [avg_acc, avg_rec, avg_f1], color='green')
            plt.title('Final Performance Metrics')
            plt.ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig('telemetry/performance_figure.png')
            print("Performance figure saved to telemetry/performance_figure.png")
    except ImportError:
        print("Note: Matplotlib not found. Skipping plot generation.")

if __name__ == "__main__":
    evaluate_model()
