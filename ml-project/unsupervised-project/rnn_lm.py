# rnn_lm.py

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem

# ========== 1) Advanced Tokenization ==========
SMILES_SPECIAL_TOKENS = [
    'Cl', 'Br', 'Si', 'Se', 'Mg', 'Zn', 'Mn', 'Na', 'Ca', 'Al'
    # Add more as needed
]

def tokenize_smiles(smiles: str):
    """
    Tokenize SMILES by matching multi-char tokens (Cl, Br, Si, etc.) first,
    then fallback to single-character tokens.
    """
    i = 0
    tokens = []
    while i < len(smiles):
        matched = False
        for special in SMILES_SPECIAL_TOKENS:
            if smiles[i:].startswith(special):
                tokens.append(special)
                i += len(special)
                matched = True
                break
        if not matched:
            tokens.append(smiles[i])
            i += 1
    return tokens


# ========== 2) Dataset ==========
class RNNUnsupervisedDataset(Dataset):
    def __init__(self, df, smiles_col="FragmentSMILES", meta_col="MetaText",
                 max_length=200, add_meta=True):
        """
        df: DataFrame with at least 'FragmentSMILES' & 'MetaText' columns.
        max_length: maximum token sequence length
        add_meta: whether to include metadata in the token sequence
        """
        self.df = df.reset_index(drop=True)
        self.smiles_col = smiles_col
        self.meta_col = meta_col
        self.max_length = max_length
        self.add_meta = add_meta
        
        self.tokenized_data = []
        for _, row in self.df.iterrows():
            smi = str(row[smiles_col]).strip()
            
            # Optional: canonicalize again if you like
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            smi_can = Chem.MolToSmiles(mol)
            
            smi_tokens = tokenize_smiles(smi_can)
            
            meta_tokens = []
            if self.add_meta and meta_col in row:
                meta_str = str(row[meta_col])
                # We'll assume meta_str is something like "<INDICATION=...> <TARGET=...>"
                meta_tokens = meta_str.strip().split()
            
            full_seq = ["<START>"] + meta_tokens + smi_tokens + ["<END>"]
            self.tokenized_data.append(full_seq)
        
        # Build vocab from all tokens
        self.build_vocab()
        
        # Encode
        self.encoded_data = []
        for tokens in self.tokenized_data:
            enc = [self.char_to_idx.get(t, self.char_to_idx["<UNK>"]) 
                   for t in tokens[:self.max_length]]
            if len(enc) < self.max_length:
                enc += [self.char_to_idx["<PAD>"]] * (self.max_length - len(enc))
            self.encoded_data.append(enc)
    
    def build_vocab(self):
        all_tokens = []
        for seq in self.tokenized_data:
            all_tokens.extend(seq)
        special_tokens = ["<START>", "<END>", "<PAD>", "<UNK>"]
        # Combine, ensure unique
        vocab_set = set(all_tokens + special_tokens)
        self.vocab = sorted(list(vocab_set))
        
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.encoded_data[idx], dtype=torch.long)


# ========== 3) Multi-layer RNN Model ==========
class MultiLayerRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, rnn_type="GRU", dropout=0.2):
        super(MultiLayerRNN, self).__init__()
        self.rnn_type = rnn_type
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        if rnn_type == "GRU":
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                              dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                               dropout=dropout, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        """
        x: (batch, seq)
        hidden: (num_layers, batch, hidden_dim) for GRU, or a tuple for LSTM
        """
        emb = self.embedding(x)  # (batch, seq, embed_dim)
        out, hidden = self.rnn(emb, hidden)  # out: (batch, seq, hidden_dim)
        logits = self.fc(out)  # (batch, seq, vocab_size)
        return logits, hidden


# ========== 4) Training Loop with Val Split & Plot ==========
def train_rnn_lm(csv_path="cancer_inhibitors_fragments.csv",
                 smiles_col="FragmentSMILES",
                 meta_col="MetaText",
                 model_save_path="rnn_lm.pt",
                 epochs=10, batch_size=32, lr=0.001):
    """
    Trains a multi-layer RNN language model with a train/val split,
    plots loss and perplexity, and saves the final model.
    """
    # 1) Load CSV
    df = pd.read_csv(csv_path)
    
    # 2) Shuffle & Split (e.g., 90% train, 10% val)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = int(0.9 * len(df))
    df_train = df.iloc[:split_idx].copy()
    df_val = df.iloc[split_idx:].copy()
    
    # 3) Create Dataset objects
    train_data = RNNUnsupervisedDataset(df_train, smiles_col=smiles_col, meta_col=meta_col)
    val_data   = RNNUnsupervisedDataset(df_val, smiles_col=smiles_col, meta_col=meta_col,
                                        max_length=train_data.max_length,
                                        add_meta=train_data.add_meta)
    
    # Ensure both share the same vocab (important!)
    val_data.vocab       = train_data.vocab
    val_data.char_to_idx = train_data.char_to_idx
    val_data.idx_to_char = train_data.idx_to_char
    
    # Encode val_data with that shared vocab
    val_data.encoded_data = []
    for seq in val_data.tokenized_data:
        enc = [val_data.char_to_idx.get(t, val_data.char_to_idx["<UNK>"]) 
               for t in seq[:train_data.max_length]]
        if len(enc) < train_data.max_length:
            enc += [val_data.char_to_idx["<PAD>"]] * (train_data.max_length - len(enc))
        val_data.encoded_data.append(enc)
    
    # 4) DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # 5) Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiLayerRNN(len(train_data.vocab), embed_dim=128, hidden_dim=256,
                          num_layers=2, rnn_type="GRU", dropout=0.2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    criterion = nn.CrossEntropyLoss(ignore_index=train_data.char_to_idx["<PAD>"])
    
    train_losses = []
    val_perplexities = []
    
    # 6) Training & Validation Loop
    for epoch in range(1, epochs+1):
        # --- TRAIN ---
        model.train()
        epoch_train_loss = 0.0
        
        for batch in train_loader:
            batch = batch.to(device)  # shape: (batch_size, seq_len)
            
            # Next-token prediction
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            optimizer.zero_grad()
            logits, _ = model(x)
            # logits: (batch_size, seq_len-1, vocab_size)
            # y     : (batch_size, seq_len-1)
            loss = criterion(logits.reshape(-1, len(train_data.vocab)), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- VALIDATION (Perplexity) ---
        model.eval()
        val_nll = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x = batch[:, :-1]
                y = batch[:, 1:]
                
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, len(train_data.vocab)), y.reshape(-1))
                
                # Convert to sum of nll
                val_nll += loss.item() * x.shape[0]  # multiply by batch size
                val_tokens += x.shape[0]
        
        avg_val_loss = val_nll / val_tokens
        val_ppl = math.exp(avg_val_loss) if avg_val_loss < 10 else float('inf')
        val_perplexities.append(val_ppl)
        
        # Print stats
        print(f"Epoch {epoch}/{epochs} - TrainLoss: {avg_train_loss:.4f} | ValPPL: {val_ppl:.2f}")
        
        # LR schedule
        scheduler.step(avg_val_loss)
    
    # 7) Plot training loss & validation perplexity
    epochs_list = range(1, epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_list, train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs_list, val_perplexities, label="Val Perplexity", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Validation Perplexity")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    
    # 8) Save Model
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": train_data.vocab,
        "char_to_idx": train_data.char_to_idx,
        "idx_to_char": train_data.idx_to_char
    }, model_save_path)
    print(f"[INFO] Model saved to {model_save_path}")
    print("[INFO] Training curves saved to training_curves.png")

if __name__ == "__main__":
    train_rnn_lm(
        csv_path="cancer_inhibitors_fragments.csv",
        smiles_col="FragmentSMILES",
        meta_col="MetaText",
        model_save_path="rnn_lm.pt",
        epochs=100,
        batch_size=32,
        lr=0.001
    )
