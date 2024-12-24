# supervised_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from unsupervised_training import (
    GraphAutoEncoder, build_graph, load_data,
    MODEL_SAVE_PATH, EMBED_DIM, AUGMENTED_CSV
)
import numpy as np
from sklearn.model_selection import train_test_split

class SimpleMLP(nn.Module):
    """
    Simple feed-forward network for binary classification
    from embeddings + optional descriptors.
    """
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def main():
    # 1. Load augmented CSV + build graph
    df = load_data(AUGMENTED_CSV)
    data = build_graph(df)
    
    # 2. Load the unsupervised GCN and compute embeddings
    in_channels = data.x.size(1)
    gcn = GraphAutoEncoder(in_channels, EMBED_DIM)
    gcn.load_state_dict(torch.load(MODEL_SAVE_PATH))
    gcn.eval()

    with torch.no_grad():
        z, _ = gcn(data)  # shape [num_nodes, EMBED_DIM]

    # 3. Suppose we have a column "InhibitorLabel" for supervised training
    #    (1 = effective inhibitor, 0 = not an inhibitor).
    if 'InhibitorLabel' not in df.columns:
        raise ValueError("DataFrame must have an 'InhibitorLabel' column for supervised training.")
    
    labels = df['InhibitorLabel'].values.astype(np.float32)  # shape [num_nodes]
    
    # 4. Optionally combine embeddings with other descriptors (e.g., MolWeight, LogP, etc.)
    # For demonstration, let's show how you could combine them:
    molwt = df['MolWeight'].fillna(0).values
    logp = df['LogP'].fillna(0).values
    extra_feats = np.column_stack([molwt, logp])

    # Our final input to the MLP is [ z | extra_feats ]
    X = np.hstack([z.cpu().numpy(), extra_feats])  # shape [num_nodes, EMBED_DIM + 2]
    y = labels  # shape [num_nodes]

    # 5. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Convert to Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float)
    y_train_t = torch.tensor(y_train, dtype=torch.float).view(-1, 1)
    X_test_t = torch.tensor(X_test, dtype=torch.float)
    y_test_t = torch.tensor(y_test, dtype=torch.float).view(-1, 1)

    # 7. Create MLP
    in_dim = X.shape[1]
    model = SimpleMLP(in_dim, hidden_dim=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 8. Train loop
    epochs = 30
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # 9. Evaluate
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_t)
        predicted_labels = (predictions >= 0.5).float()
        accuracy = (predicted_labels.eq(y_test_t).sum() / y_test_t.size(0)).item()
    print(f"Test Accuracy: {accuracy:.4f}")

    # Optionally save the supervised model
    torch.save(model.state_dict(), "supervised_mlp_model.pt")
    print("Supervised MLP model saved.")

if __name__ == "__main__":
    main()
