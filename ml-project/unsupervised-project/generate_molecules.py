# generate_molecules.py

import torch
import numpy as np
import pandas as pd

from rnn_lm import MultiLayerRNN  # import model class
import math

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = MultiLayerRNN(
        vocab_size=len(checkpoint["vocab"]),
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        rnn_type="GRU",
        dropout=0.2
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    vocab       = checkpoint["vocab"]
    char_to_idx = checkpoint["char_to_idx"]
    idx_to_char = checkpoint["idx_to_char"]
    
    return model, vocab, char_to_idx, idx_to_char

def sample_sequence(model, vocab, char_to_idx, idx_to_char,
                    max_length=200, temperature=1.0):
    """
    Sample a sequence from the trained RNN, starting with <START> and
    ending at <END> or max_length.
    """
    device = next(model.parameters()).device
    start_token = "<START>"
    end_token   = "<END>"
    
    x = torch.tensor([[char_to_idx[start_token]]], dtype=torch.long, device=device)
    hidden = None
    
    generated_tokens = []
    for _ in range(max_length):
        logits, hidden = model(x, hidden)  # logits shape: (1, 1, vocab_size)
        logits = logits[0, -1, :] / temperature
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy()
        
        next_idx = np.random.choice(len(probs), p=probs)
        next_token = idx_to_char[next_idx]
        if next_token == end_token:
            break
        
        generated_tokens.append(next_token)
        x = torch.tensor([[next_idx]], dtype=torch.long, device=device)
    
    return "".join(token if len(token) > 1 else token for token in generated_tokens)

if __name__ == "__main__":
    model, vocab, c2i, i2c = load_model("rnn_lm.pt")
    model = model.cpu()  # place model on CPU for sampling
    
    # Generate 10 sequences
    for i in range(10):
        seq = sample_sequence(model, vocab, c2i, i2c, max_length=150, temperature=0.8)
        print(f"Generated #{i+1}:\n{seq}\n")
