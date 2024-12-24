# unsupervised_testing.py
import torch
import torch.nn.functional as F
from unsupervised_training import (
    GraphAutoEncoder, build_graph, load_data,
    MODEL_SAVE_PATH, EMBED_DIM, AUGMENTED_CSV
)

def main():
    # 1. Load augmented dataset and build graph
    df = load_data(AUGMENTED_CSV)
    data = build_graph(df)

    # 2. Create same model architecture
    in_channels = data.x.size(1)
    model = GraphAutoEncoder(in_channels, EMBED_DIM)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # 3. Inference: Generate embeddings
    with torch.no_grad():
        z, _ = model(data)  # z has shape [num_nodes, EMBED_DIM]

    print("Learned node embeddings shape:", z.shape)

    # Reconstruct full adjacency if needed
    adjacency_recon = model.reconstruct_all(z)
    print("Reconstructed adjacency shape:", adjacency_recon.shape)

    # Similarity example
    for i in range(321):
        for j in range(i+1, 321):
            sim = F.cosine_similarity(z[i].unsqueeze(0), z[j].unsqueeze(0))
            print(f"Similarity between node {i} and {j}: {sim.item():.4f}")

if __name__ == "__main__":
    main()
