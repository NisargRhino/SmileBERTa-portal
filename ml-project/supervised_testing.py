# supervised_testing.py

import torch
import numpy as np
from unsupervised_training import (
    GraphAutoEncoder, build_graph, load_data,
    MODEL_SAVE_PATH, EMBED_DIM, AUGMENTED_CSV
)
from supervised_training import SimpleMLP
import pandas as pd

def main():
    # 1. Load data & GCN
    df = load_data(AUGMENTED_CSV)
    data = build_graph(df)

    gcn = GraphAutoEncoder(in_channels=data.x.size(1), hidden_dim=EMBED_DIM)
    gcn.load_state_dict(torch.load(MODEL_SAVE_PATH))
    gcn.eval()

    # 2. Get embeddings
    with torch.no_grad():
        z, _ = gcn(data)

    # 3. Load MLP
    in_dim = EMBED_DIM + 2  # EMBED_DIM + # of extra features used
    mlp = SimpleMLP(in_dim)
    mlp.load_state_dict(torch.load("supervised_mlp_model.pt"))
    mlp.eval()

    # 4. Prepare input
    molwt = df['MolWeight'].fillna(0).values
    logp = df['LogP'].fillna(0).values
    extra_feats = np.column_stack([molwt, logp])

    X = np.hstack([z.cpu().numpy(), extra_feats])  # shape [num_nodes, in_dim]

    # 5. Predict
    X_t = torch.tensor(X, dtype=torch.float)
    with torch.no_grad():
        preds = mlp(X_t)
        predicted_labels = (preds >= 0.5).float().view(-1)

    # 6. Print out first few predictions
    print("Predicted Inhibitor Labels for first 5 drugs:")
    for i in range(min(5, len(predicted_labels))):
        print(f"Drug {i}: Prediction = {predicted_labels[i].item()}")

if __name__ == "__main__":
    main()
