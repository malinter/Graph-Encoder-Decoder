import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling
from models.ged_core import GEDCore
import os
import csv
from datetime import datetime

class SpectrumTrainer:
    """
    Trains the Graph Encoder-Decoder with GAT and Negative Sampling.
    """
    def __init__(self, model, lr=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss() # Better for numerical stability with logits

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for data in loader:
            self.optimizer.zero_grad()
            
            # 1. Encode (Node Embeddings)
            out = self.model(data.x, data.edge_index, data.batch)
            node_z = out[0] if isinstance(out, tuple) else out
            
            # 2. Positive Sampling (Real Edges)
            pos_edge_index = data.edge_index
            pos_out = self.model.decoder(node_z, pos_edge_index)
            pos_loss = self.criterion(pos_out, torch.ones(pos_out.size(0)))
            
            # 3. Negative Sampling (Forbidden Edges)
            neg_edge_index = negative_sampling(
                edge_index=data.edge_index,
                num_nodes=data.x.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
            neg_out = self.model.decoder(node_z, neg_edge_index)
            neg_loss = self.criterion(neg_out, torch.zeros(neg_out.size(0)))
            
            # 4. Total Loss
            loss = pos_loss + neg_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / len(loader)

def count_parameters(model):
    """Calculates total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    os.makedirs('telemetry', exist_ok=True)
    log_file = 'telemetry/training_log.csv'
    
    if not os.path.exists('synthetic_system_graphs.pt'):
        print("Error: synthetic_system_graphs.pt not found.")
        return
    
    dataset = torch.load('synthetic_system_graphs.pt', weights_only=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # 1. Initialize Model with GAT and High Capacity
    input_dim = dataset[0].x.size(1)
    model = GEDCore(input_dim=input_dim, hidden_dim=128, latent_dim=32, use_gat=True)
    
    total_params = count_parameters(model)
    print(f"--- Model Brain Weight (GAT) ---")
    print(f"Total Trainable Parameters: {total_params:,}")
    print(f"--------------------------------")
    
    trainer = SpectrumTrainer(model)
    
    # 2. Training Loop
    print("Starting Focused training on System Spectrum (1,000 Epochs)...")
    epochs = 1000
    
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Epoch', 'Loss'])
        
        for epoch in range(1, epochs + 1):
            loss = trainer.train_epoch(loader)
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            writer.writerow([timestamp, epoch, loss])
            
            if epoch % 10 == 0:
                f.flush()
            
            # Periodic model checkpointing
            if epoch % 100 == 0:
                os.makedirs('weights', exist_ok=True)
                model.save_weights('weights/ged_foundation.pth')
            
            if epoch % 50 == 0 or epoch == 1:
                print(f"{timestamp} Epoch {epoch:04d}, Loss: {loss:.4f}")

    os.makedirs('weights', exist_ok=True)
    model.save_weights('weights/ged_foundation.pth')
    print(f"Training complete. Weights saved to weights/ged_foundation.pth")

if __name__ == "__main__":
    main()
