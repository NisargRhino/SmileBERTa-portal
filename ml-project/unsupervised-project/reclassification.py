# reclassification.py

import math
import torch
import torch.nn.functional as F
import pandas as pd

from rnn_lm import MultiLayerRNN

def load_model_for_eval(model_path):
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
    
    return model, checkpoint["vocab"], checkpoint["char_to_idx"], checkpoint["idx_to_char"]

def compute_sequence_perplexity(model, vocab, char_to_idx, idx_to_char, sequence_tokens):
    """
    Given a tokenized sequence (including <START> and <END>),
    compute perplexity under the model.
    """
    device = next(model.parameters()).device
    
    # Convert tokens -> indices
    idxs = [char_to_idx.get(t, char_to_idx["<UNK>"]) for t in sequence_tokens]
    # Convert to tensor, shape (1, seq_len)
    x = torch.tensor([idxs], dtype=torch.long, device=device)
    
    if x.size(1) < 2:
        return float('inf')
    
    # Next-token prediction
    x_input  = x[:, :-1]
    x_target = x[:, 1:]
    
    with torch.no_grad():
        logits, _ = model(x_input)
        # logits shape = (1, seq_len-1, vocab_size)
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size),
                               x_target.reshape(-1),
                               reduction='sum')
        nll = loss.item()
        num_tokens = x_target.numel()
        ppl = math.exp(nll / num_tokens)
        return ppl

if __name__ == "__main__":
    model, vocab, c2i, i2c = load_model_for_eval("rnn_lm.pt")
    model = model.cpu()
    
    # Example: Suppose we have some generated sequences or want to test some SMILES
    # Let's define them as a list of tokens, including <START> and <END>
    test_sequences = [
        ["<START>", "C", "C", "N", "<END>"],
        ["<START>", "<INDICATION=BreastCancer>", "C", "l", "Br", "<END>"]
    ]
    
    for seq_tokens in test_sequences:
        ppl = compute_sequence_perplexity(model, vocab, c2i, i2c, seq_tokens)
        print(f"Sequence: {seq_tokens}\nPerplexity = {ppl:.2f}\n")
