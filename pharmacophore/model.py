"""The pocket-conditioned per-atom interaction model.

A set transformer over a ligand's atoms, conditioned on a pocket descriptor,
predicting for each heavy atom which typed interactions it makes. Self-attention
across atoms is what lets the model see an atom in the context of the rest of the
molecule, which is most of what an atom descriptor cannot express on its own.

Two deliberate choices:

**The pocket enters as a conditioning vector, not as a concatenated feature.**
It is projected and added to every atom's representation, so removing it is a
single flag rather than a different architecture — which is what makes the
no-pocket ablation an honest comparison instead of two unrelated models.

**The output is per atom, over the input's own atoms.** There is no generation
anywhere, so a predicted pharmacophore is a substructure of the input by
construction and the hallucination rate that sank the old fragment model is
zero rather than measured.

Nothing here needs a GPU to be correct, only to be fast. It runs on CPU, MPS or
CUDA unchanged.
"""
import torch
import torch.nn as nn

from pharmacophore.interactions import INTERACTIONS


def device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class PocketConditionedAtomModel(nn.Module):
    """atoms (+ pocket) -> one probability per interaction type per atom."""

    def __init__(self, atom_dim, pocket_dim, hidden=192, layers=4, heads=6,
                 dropout=0.1, n_types=len(INTERACTIONS), use_pocket=True):
        super().__init__()
        self.use_pocket = use_pocket
        self.atom_in = nn.Sequential(
            nn.Linear(atom_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        # Projected to the same width as the atom representation so it can be
        # added rather than concatenated: the ablation then changes one term,
        # not the input dimensionality.
        self.pocket_in = nn.Sequential(
            nn.Linear(pocket_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_types))

    def forward(self, atoms, pocket, mask):
        """atoms (B, A, atom_dim), pocket (B, pocket_dim), mask (B, A) with True = real atom."""
        h = self.atom_in(atoms)
        if self.use_pocket:
            h = h + self.pocket_in(pocket).unsqueeze(1)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(h)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GeometricPocketAtomModel(nn.Module):
    """Atoms attend to their own 3D pocket neighbourhoods.

    The earlier model projected a six-number pocket descriptor and added it to
    every atom alike. Measured across three independent cycles, that cost
    -0.013 to +0.003 macro AUC against having no pocket at all, while a random
    forest on the same features gained +0.040 — the information was there and
    the conditioning was throwing it away. A vector added identically to every
    atom can say "this pocket is 11% anionic"; it cannot say "*this* atom has a
    carboxylate oxygen 2.9 A away", which is the fact that decides the label.

    Here each atom cross-attends over its own nearest pocket atoms, described
    by distance and chemistry. Two properties are deliberate:

    **The ablation is still one flag.** `use_pocket=False` masks the
    neighbourhood instead of changing the architecture, so the control has the
    same parameters, the same depth and the same training schedule. That is
    what made the earlier comparison trustworthy and it would have been easy to
    lose here.

    **Only distances enter, never coordinates.** The representation is
    invariant to rotation and translation by construction, so nothing is spent
    learning that a PDB file's reference frame is arbitrary.
    """

    def __init__(self, atom_dim, neighbour_dim, hidden=192, layers=4, heads=6,
                 dropout=0.1, n_types=len(INTERACTIONS), use_pocket=True):
        super().__init__()
        self.use_pocket = use_pocket
        self.hidden = hidden
        self.atom_in = nn.Sequential(
            nn.Linear(atom_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.neighbour_in = nn.Sequential(
            nn.Linear(neighbour_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.pocket_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.pocket_norm = nn.LayerNorm(hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_types))

    def forward(self, atoms, neighbours, neighbour_mask, mask):
        """atoms (B, A, atom_dim), neighbours (B, A, K, nd), masks (B, A, K) and (B, A)."""
        h = self.atom_in(atoms)
        if self.use_pocket:
            batch, n_atoms, k, _ = neighbours.shape
            queries = h.reshape(batch * n_atoms, 1, self.hidden)
            keys = self.neighbour_in(neighbours).reshape(batch * n_atoms, k, self.hidden)
            padding = ~neighbour_mask.reshape(batch * n_atoms, k)
            # A padded ligand slot has no neighbours at all, and attention over
            # an entirely masked row softmaxes to NaN and poisons the batch.
            # Those rows are given a dummy key and their output discarded.
            empty = padding.all(dim=1)
            padding = padding.clone()
            padding[empty, 0] = False
            context, _weights = self.pocket_attention(
                queries, keys, keys, key_padding_mask=padding, need_weights=False)
            context = context.reshape(batch, n_atoms, self.hidden)
            context = context * (~empty).reshape(batch, n_atoms, 1)
            h = self.pocket_norm(h + context)
        h = self.encoder(h, src_key_padding_mask=~mask)
        return self.head(h)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def collate_geometric(batch, atom_dim, neighbour_dim, k):
    """Pad a batch for the geometric model: atoms, neighbourhoods and two masks."""
    width = max(e['n_heavy'] for e in batch)
    atoms = torch.zeros(len(batch), width, atom_dim)
    neighbours = torch.zeros(len(batch), width, k, neighbour_dim)
    neighbour_mask = torch.zeros(len(batch), width, k, dtype=torch.bool)
    labels = torch.zeros(len(batch), width, len(INTERACTIONS))
    mask = torch.zeros(len(batch), width, dtype=torch.bool)
    for i, example in enumerate(batch):
        n = example['n_heavy']
        atoms[i, :n] = torch.from_numpy(example['atom_features'][:n])
        labels[i, :n] = torch.from_numpy(example['labels'][:n].astype('float32'))
        mask[i, :n] = True
        nf, nm = example.get('neighbour_features'), example.get('neighbour_mask')
        if nf is not None:
            rows = min(n, nf.shape[0])
            neighbours[i, :rows] = torch.from_numpy(nf[:rows])
            neighbour_mask[i, :rows] = torch.from_numpy(nm[:rows])
    return atoms, neighbours, neighbour_mask, mask, labels


def collate(batch, atom_dim, pocket_dim):
    """Pad a batch of variable-length molecules; mask marks the real atoms."""
    width = max(e['n_heavy'] for e in batch)
    atoms = torch.zeros(len(batch), width, atom_dim)
    labels = torch.zeros(len(batch), width, len(INTERACTIONS))
    mask = torch.zeros(len(batch), width, dtype=torch.bool)
    pocket = torch.zeros(len(batch), pocket_dim)
    for i, example in enumerate(batch):
        n = example['n_heavy']
        atoms[i, :n] = torch.from_numpy(example['atom_features'][:n])
        labels[i, :n] = torch.from_numpy(example['labels'][:n].astype('float32'))
        mask[i, :n] = True
        pocket[i] = torch.from_numpy(example['pocket_features'])
    return atoms, pocket, mask, labels


def positive_weights(examples, cap=50.0):
    """Per-type pos_weight for BCE, since the labels are heavily imbalanced.

    Capped: the rarest types occur in well under 1% of atoms, and an uncapped
    inverse-frequency weight there makes training diverge rather than attentive.
    """
    import numpy as np
    labels = np.vstack([e['labels'] for e in examples])
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = np.where(positives > 0, negatives / np.maximum(positives, 1), 1.0)
    return torch.tensor(np.minimum(weights, cap), dtype=torch.float)


def masked_bce(logits, labels, mask, pos_weight=None):
    """Loss over real atoms only; padding must not contribute."""
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, reduction='none', pos_weight=pos_weight)
    loss = loss * mask.unsqueeze(-1)
    return loss.sum() / mask.sum().clamp(min=1) / labels.shape[-1]
