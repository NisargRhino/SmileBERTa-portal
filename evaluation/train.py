"""Retraining, learning curves, data-scaling curves and the overfitting interventions.

Why retraining is not optional. The shipped checkpoints were trained on an
unknown split, so any test set defined afterwards almost certainly contains
molecules they were trained on. Evaluating them tells you what today's users
get, but it cannot tell you how the architecture generalises. Only a model
trained on the scaffold-split *training* set and measured on the scaffold-split
*test* set can do that, so that is what this file builds.

The recipe reproduces the deployed one rather than improving on it: a
`RobertaForMaskedLM` initialised from `seyonec/ChemBERTa-zinc-base-v1`, fed the
SMILES as input and the answer string as the label at every position, decoded
with argmax at inference. Swapping in a sequence-classification head would very
likely score better and would also make the numbers incomparable with the models
actually in production, which is the comparison this benchmark exists to make.

Interventions are applied one at a time, in the order the spec sets out, each
measured on its own. Nothing here touches the test set: every decision is made
on validation loss.
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaForMaskedLM, RobertaTokenizer

from evaluation import data as D
from evaluation.baselines import partition
from evaluation.splits import SPLIT_DIR, load_split

BASE_MODEL = 'seyonec/ChemBERTa-zinc-base-v1'
MAX_LENGTH = 128
SEED = 20261126
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
CHECKPOINTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device():
    return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')


class PairDataset(Dataset):
    """(input SMILES -> answer string) pairs.

    The label tensor is the tokenised answer with padding masked out to -100, so
    loss is taken only over real answer tokens. Padding positions would otherwise
    dominate: an answer like "cns" is three tokens inside a 128-wide window.

    Sequences come back unpadded and `collate` pads each batch to its own longest
    member. Padding every SMILES to 128 instead costs about half the training
    time on this dataset and changes nothing: pad positions are excluded by the
    attention mask and pad labels by the -100, so the loss is identical either
    way. Inference still pads to a fixed 128, because that is what the deployed
    code does.
    """

    def __init__(self, pairs, tokenizer, weights=None):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.weights = weights

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        source, target = self.pairs[i]
        enc = self.tokenizer(source, max_length=MAX_LENGTH, truncation=True)
        lab = self.tokenizer(target, max_length=MAX_LENGTH, truncation=True)
        item = {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
                'labels': lab['input_ids']}
        if self.weights is not None:
            item['weight'] = float(self.weights[i])
        return item


def collate(batch, pad_id=1):
    """Pad a batch to its longest sequence; labels are padded with -100.

    Input and label sequences are padded to the same width because this recipe
    reads the answer off the MLM head position by position, so the two tensors
    have to line up.
    """
    width = max(max(len(b['input_ids']), len(b['labels'])) for b in batch)
    out = {'input_ids': [], 'attention_mask': [], 'labels': []}
    weights = []
    for b in batch:
        n = len(b['input_ids'])
        out['input_ids'].append(b['input_ids'] + [pad_id] * (width - n))
        out['attention_mask'].append(b['attention_mask'] + [0] * (width - n))
        m = len(b['labels'])
        out['labels'].append(b['labels'] + [-100] * (width - m))
        if 'weight' in b:
            weights.append(b['weight'])
    tensors = {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}
    if weights:
        tensors['weight'] = torch.tensor(weights, dtype=torch.float)
    return tensors


def classification_pairs(records):
    return [(r['smiles'], ' '.join(r['labels'])) for r in records]


def fragment_pairs(records):
    """One training pair per (molecule, fragment), as the original data file is shaped.

    A molecule with six BRICS fragments becomes six pairs. The model can only
    emit one answer, so at inference it collapses to whichever it learned best —
    and the evaluation counts a hit against any of the molecule's fragments.
    """
    return [(r['smiles'], frag) for r in records for frag in r['fragments']]


def class_weights(records):
    """Inverse-frequency weight per compound, for the long tail.

    Urological has 35 training compounds against antiinfective's 2,339. Without
    weighting the model can ignore the tail entirely and still look fine on any
    average that is not macro.
    """
    counts = {}
    for rec in records:
        for lab in rec['labels']:
            counts[lab] = counts.get(lab, 0) + 1
    total = sum(counts.values())
    out = []
    for rec in records:
        w = max(total / (len(counts) * counts[lab]) for lab in rec['labels'])
        out.append(min(w, 20.0))   # capped: a 60x weight makes training unstable
    return out


def make_loader(pairs, tokenizer, batch_size, shuffle, weights=None):
    pad_id = tokenizer.pad_token_id
    return DataLoader(PairDataset(pairs, tokenizer, weights), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0,
                      collate_fn=lambda b: collate(b, pad_id))


def run_epoch(model, loader, optimiser=None, weighted=False):
    train = optimiser is not None
    model.train() if train else model.eval()
    dev = device()
    total, n_batches = 0.0, 0
    for batch in loader:
        weight = batch.pop('weight', None)
        batch = {k: v.to(dev) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            out = model(**batch)
            loss = out.loss
            if weighted and weight is not None:
                # Per-sample reweighting of the mean loss. Cheap, and enough to
                # tell whether weighting the tail helps at all.
                loss = loss * weight.to(dev).mean()
        if train:
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


def freeze_lower_layers(model, keep_top=2):
    """Freeze embeddings and all but the top `keep_top` encoder layers.

    With a few thousand examples, fine-tuning 44 M parameters end to end is a
    good way to memorise the training set; updating only the top of the stack is
    the standard answer.
    """
    for p in model.roberta.embeddings.parameters():
        p.requires_grad = False
    layers = model.roberta.encoder.layer
    for layer in layers[:max(len(layers) - keep_top, 0)]:
        for p in layer.parameters():
            p.requires_grad = False
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(train_pairs, val_pairs, *, epochs=6, batch_size=16, lr=5e-5,
                dropout=None, weight_decay=0.0, freeze_below_top=None,
                weights=None, early_stopping=True, patience=2, seed=SEED, log=print):
    """Train and return (best model, history). Model selection is on validation loss only."""
    set_seed(seed)
    tokenizer = RobertaTokenizer.from_pretrained(BASE_MODEL)
    kwargs = {}
    if dropout is not None:
        kwargs.update(hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
    model = RobertaForMaskedLM.from_pretrained(BASE_MODEL, **kwargs).to(device())
    trainable = sum(p.numel() for p in model.parameters())
    if freeze_below_top is not None:
        trainable = freeze_lower_layers(model, freeze_below_top)

    train_loader = make_loader(train_pairs, tokenizer, batch_size, True, weights)
    val_loader = make_loader(val_pairs, tokenizer, batch_size, False)
    optimiser = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=lr, weight_decay=weight_decay)

    history, best = [], {'val_loss': float('inf'), 'epoch': -1, 'state': None}
    bad_epochs = 0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, optimiser, weighted=weights is not None)
        val_loss = run_epoch(model, val_loader)
        history.append({'epoch': epoch + 1, 'train_loss': round(train_loss, 4),
                        'val_loss': round(val_loss, 4), 'seconds': round(time.time() - t0, 1)})
        log('    epoch %d  train %.4f  val %.4f  (%.0fs)'
            % (epoch + 1, train_loss, val_loss, time.time() - t0))
        if val_loss < best['val_loss'] - 1e-4:
            best = {'val_loss': val_loss, 'epoch': epoch + 1,
                    'state': copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if early_stopping and bad_epochs >= patience:
                log('    early stop: validation loss has not improved for %d epochs' % patience)
                break

    if best['state'] is not None:
        model.load_state_dict(best['state'])
    return model, {'history': history, 'best_epoch': best['epoch'],
                   'best_val_loss': round(best['val_loss'], 4),
                   'trainable_parameters': trainable}


def save(model, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    RobertaTokenizer.from_pretrained(BASE_MODEL).save_pretrained(path)
    return path


def load_task(task, split_kind):
    records = (D.load_classification() if task == 'classification' else D.load_fragments())[0]
    split_map = load_split(os.path.join(SPLIT_DIR, f'{task}_{split_kind}.csv'))
    return partition(records, split_map)


def pairs_for(task, records):
    return classification_pairs(records) if task == 'classification' else fragment_pairs(records)
