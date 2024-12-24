# unsupervised_training.py
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
import os

##############################################################################
# 1. Hyperparameters
##############################################################################
EPOCHS = 50
LR = 0.01
EMBED_DIM = 16
MODEL_SAVE_PATH = "unsupervised_gcn_model.pt"
AUGMENTED_CSV = "cancer-inhibitors-augmented.csv"

##############################################################################
# 2. Prepare your dataset
##############################################################################
def load_data(csv_path=AUGMENTED_CSV):
    df = pd.read_csv(csv_path)
    return df

def build_graph(df):
    """
    Constructs a graph from the augmented DataFrame:
      - Node features: Year, Generic, MolWeight, LogP, NumHDonors, NumHAcceptors, etc.
      - Edges: if drugs share at least one target.
    """
    # 1) Build adjacency from 'Targets'
    targets_list = []
    for i, row in df.iterrows():
        targets = str(row['Targets']).split(';')
        targets = [t.strip() for t in targets if t.strip()]
        targets_list.append(set(targets))

    num_drugs = len(df)
    row_indices = []
    col_indices = []
    for i in range(num_drugs):
        for j in range(i+1, num_drugs):
            if len(targets_list[i].intersection(targets_list[j])) > 0:
                row_indices.append(i)
                col_indices.append(j)
                row_indices.append(j)
                col_indices.append(i)

    # 2) Create adjacency matrix
    data_values = np.ones(len(row_indices))
    adjacency_coo = coo_matrix(
        (data_values, (row_indices, col_indices)),
        shape=(num_drugs, num_drugs)
    )
    edge_index, _ = from_scipy_sparse_matrix(adjacency_coo)

    # 3) Create node features
    # We'll combine numeric columns into one feature matrix
    years = df['Year'].fillna(0).astype(float).values
    years_norm = (years - years.min()) / (years.max() - years.min() + 1e-9)

    generic = df['Generic'].fillna('N').apply(lambda x: 1.0 if x.strip().upper() == 'Y' else 0.0).values
    
    molwt = df['MolWeight'].fillna(0).values
    logp = df['LogP'].fillna(0).values
    hdonors = df['NumHDonors'].fillna(0).values
    hacceptors = df['NumHAcceptors'].fillna(0).values

    # You can add more if you like (e.g., one-hot encoding of FDA, etc.)
    # Let's just stack them:
    node_features = np.column_stack([
        years_norm,
        generic,
        molwt,
        logp,
        hdonors,
        hacceptors
    ])
    
    x_tensor = torch.tensor(node_features, dtype=torch.float)
    data = Data(x=x_tensor, edge_index=edge_index)
    return data

##############################################################################
# 3. Define the Model (Graph Autoencoder)
##############################################################################
class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv2 = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

class GraphAutoEncoder(torch.nn.Module):
    """
    A simple Graph Autoencoder:
      - Encoder: GCN layers produce node embeddings.
      - Decoder: Dot-product between node embeddings to reconstruct adjacency.
    """
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim)

    def forward(self, data):
        z = self.encoder(data.x, data.edge_index)
        # Reconstruct for positive edges
        edge_i, edge_j = data.edge_index
        reconstructions = (z[edge_i] * z[edge_j]).sum(dim=1)
        return z, reconstructions

    def reconstruct_all(self, z):
        # Reconstruct full adjacency if needed
        return torch.sigmoid(torch.mm(z, z.t()))

##############################################################################
# 4. Training loop
##############################################################################
def train_model(data, model, epochs=EPOCHS, lr=LR):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    pos_i, pos_j = data.edge_index

    for epoch in range(epochs):
        optimizer.zero_grad()
        z, recon = model(data)
        
        # All existing edges are "positive"
        pos_label = torch.ones(pos_i.size(0), dtype=torch.float)

        recon_sigmoid = torch.sigmoid(recon)
        loss_pos = F.binary_cross_entropy(recon_sigmoid, pos_label)
        
        # No negative edges in this example for brevity
        loss = loss_pos
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

def main():
    # 1. Load augmented data
    df = load_data(AUGMENTED_CSV)
    data = build_graph(df)

    # 2. Create Model
    in_channels = data.x.size(1)  # number of input features
    model = GraphAutoEncoder(in_channels, EMBED_DIM)

    # 3. Train
    train_model(data, model, EPOCHS, LR)

    # 4. Save
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Unsupervised GCN model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
