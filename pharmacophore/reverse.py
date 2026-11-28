"""Direction 2: a pocket in, the fragments that bind it out.

The per-atom model cannot do this. It scores each ligand atom from its distances
to pocket atoms, which means it needs a pose, and obtaining a pose for a
candidate fragment means docking it -- the unreliable step this project exists
to avoid. Asking the forward model to screen fragments is circular.

So this is a separate, deliberately pose-free model: a pocket descriptor in, a
probability per fragment out, trained on which fragments were actually found
bound in each complex. Nothing here needs coordinates, so unlike the forward
model it can in principle score a pocket that has never had a ligand in it --
subject to the apo caveat in the README, since the pockets it is *trained* on
were still defined by a bound ligand.

Evaluated the way the spec demands: recall@10, @50 and @100 against the
fragments actually observed, with the vocabulary size stated, because recall
without a denominator means nothing. The baseline to beat is the global
frequency prior -- "always answer benzene" is a real and surprisingly strong
strategy, and `BENCHMARK.md` is the record of it beating a 44M-parameter
transformer.
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold

from pharmacophore import dataset as D
from pharmacophore.splits import REGIMES, load_split

RDLogger.DisableLog('rdApp.*')

SEED = 20261126
# A fragment must appear in at least this many training complexes to enter the
# vocabulary. Singletons are unlearnable and would inflate the denominator
# while making recall look worse for no informative reason.
MIN_FRAGMENT_COUNT = 20
TOP_K = (10, 50, 100)

AMINO_ACIDS = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
               'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
               'TYR', 'VAL']


def pocket_profile(record):
    """A pose-free pocket descriptor: residue composition plus size.

    Richer than the six fractions the forward model uses, because here the
    pocket is the *only* input -- there is no ligand to carry the rest of the
    signal. Still computable from any defined pocket without a bound ligand.
    """
    residues = [r.split()[0] for r in record.get('pocket_residues', []) if r.split()]
    counts = Counter(residues)
    total = max(len(residues), 1)
    profile = [counts.get(aa, 0) / total for aa in AMINO_ACIDS]
    # Size enters as a log, not a raw count: the difference between a 10- and a
    # 20-residue site is real, between 60 and 70 much less so.
    profile.append(np.log1p(len(residues)) / 5.0)
    return np.array(profile, dtype=np.float32)


def fragments_of(smiles, scheme='brics'):
    """The fragment vocabulary entries present in one ligand."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    out = set()
    if scheme in ('brics', 'both'):
        try:
            for frag in BRICS.BRICSDecompose(mol, returnMols=False):
                # Strip the attachment-point isotope labels so the same
                # chemical fragment from two different cut sites is one entry.
                cleaned = Chem.MolFromSmiles(frag)
                if cleaned is None:
                    continue
                for atom in cleaned.GetAtoms():
                    atom.SetIsotope(0)
                smi = Chem.MolToSmiles(cleaned).replace('*', '[H]')
                smi = Chem.CanonSmiles(smi) if smi else ''
                if smi and 3 <= Chem.MolFromSmiles(smi).GetNumHeavyAtoms() <= 20:
                    out.add(smi)
        except Exception:
            pass
    if scheme in ('scaffold', 'both'):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
            if scaffold:
                out.add(scaffold)
        except Exception:
            pass
    return out


def build_vocabulary(records, scheme='brics', min_count=MIN_FRAGMENT_COUNT, log=print):
    counts = Counter()
    per_record = []
    for record in records:
        frags = fragments_of(record['smiles'], scheme)
        per_record.append(frags)
        counts.update(frags)
    vocabulary = sorted(f for f, n in counts.items() if n >= min_count)
    log('  vocabulary: %d fragments seen >= %d times (of %d distinct)'
        % (len(vocabulary), min_count, len(counts)))
    return vocabulary, per_record, counts


def _matrix(records, per_record, vocabulary):
    index = {f: i for i, f in enumerate(vocabulary)}
    x = np.stack([pocket_profile(r) for r in records])
    y = np.zeros((len(records), len(vocabulary)), dtype=np.float32)
    for row, frags in enumerate(per_record):
        for f in frags:
            if f in index:
                y[row, index[f]] = 1.0
    return x, y


def recall_at_k(y_true, scores, ks=TOP_K):
    """Fraction of a complex's true fragments that appear in the top k.

    Averaged over complexes that have at least one true fragment: a complex
    whose ligand contributed nothing to the vocabulary cannot be scored and
    including it as a zero would punish the model for the vocabulary's cutoff.
    """
    out = {}
    usable = y_true.sum(axis=1) > 0
    if not usable.any():
        return {f'recall@{k}': None for k in ks}
    for k in ks:
        top = np.argsort(-scores, axis=1)[:, :k]
        hits = []
        for row in np.where(usable)[0]:
            true_set = set(np.where(y_true[row])[0])
            hits.append(len(true_set & set(top[row])) / len(true_set))
        out[f'recall@{k}'] = round(float(np.mean(hits)), 4)
    out['scored_complexes'] = int(usable.sum())
    return out


def frequency_prior(y_train, n_test):
    """Rank every fragment by how often it was seen, ignoring the pocket."""
    rates = y_train.mean(axis=0)
    return np.tile(rates, (n_test, 1))


def nearest_pocket(x_train, y_train, x_test, k=25):
    """Retrieval: what bound the most similar pockets in training.

    No learning at all. Included because a retrieval baseline is the honest
    thing to beat before claiming a learned model adds anything.
    """
    train_norm = x_train / (np.linalg.norm(x_train, axis=1, keepdims=True) + 1e-8)
    test_norm = x_test / (np.linalg.norm(x_test, axis=1, keepdims=True) + 1e-8)
    scores = np.zeros((x_test.shape[0], y_train.shape[1]), dtype=np.float32)
    chunk = 512
    for start in range(0, test_norm.shape[0], chunk):
        sim = test_norm[start:start + chunk] @ train_norm.T
        neighbours = np.argsort(-sim, axis=1)[:, :k]
        for i, row in enumerate(neighbours):
            scores[start + i] = y_train[row].mean(axis=0)
    return scores


def learned(x_train, y_train, x_test, epochs=60, hidden=256, log=print):
    """A small MLP from pocket composition to fragment probabilities."""
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    dev = torch.device('cpu')          # tiny model; MPS transfer costs more
    net = nn.Sequential(
        nn.Linear(x_train.shape[1], hidden), nn.GELU(), nn.LayerNorm(hidden),
        nn.Dropout(0.1),
        nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        nn.Linear(hidden, y_train.shape[1])).to(dev)
    optimiser = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=0.01)
    xt = torch.from_numpy(x_train).to(dev)
    yt = torch.from_numpy(y_train).to(dev)
    loss_fn = nn.BCEWithLogitsLoss()
    batch = 256
    for epoch in range(epochs):
        net.train()
        order = torch.randperm(xt.shape[0])
        total = 0.0
        for start in range(0, xt.shape[0], batch):
            idx = order[start:start + batch]
            loss = loss_fn(net(xt[idx]), yt[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
        if (epoch + 1) % 20 == 0:
            log('    epoch %2d  loss %.4f' % (epoch + 1, total / max(1, xt.shape[0] // batch)))
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(x_test).to(dev))).cpu().numpy()


def run(release, regime, scheme='brics', log=print):
    split_map = load_split(release, regime)
    train_records, test_records = [], []
    for record in D.iter_records(release):
        if record.get('rejected'):
            continue
        where = split_map.get((record['pdb_id'], record['comp_id']))
        if where == 'train':
            train_records.append(record)
        elif where == 'test':
            test_records.append(record)
    log('[%s] %d train / %d test complexes' % (regime, len(train_records), len(test_records)))
    if len(train_records) < 50 or len(test_records) < 10:
        return {'regime': regime, 'error': 'not enough complexes in this regime'}

    vocabulary, train_frags, counts = build_vocabulary(train_records, scheme, log=log)
    if not vocabulary:
        return {'regime': regime, 'error': 'empty fragment vocabulary'}
    test_frags = [fragments_of(r['smiles'], scheme) for r in test_records]

    x_train, y_train = _matrix(train_records, train_frags, vocabulary)
    x_test, y_test = _matrix(test_records, test_frags, vocabulary)
    coverage = float((y_test.sum(axis=1) > 0).mean())
    log('  %.1f%% of test complexes have at least one in-vocabulary fragment'
        % (100 * coverage))

    out = {'regime': regime, 'scheme': scheme,
           'vocabulary_size': len(vocabulary),
           'train_complexes': len(train_records),
           'test_complexes': len(test_records),
           'test_coverage': round(coverage, 4),
           'most_common': [f for f, _ in counts.most_common(10)]}

    out['frequency_prior'] = recall_at_k(y_test, frequency_prior(y_train, len(test_records)))
    log('  frequency prior   %s' % out['frequency_prior'])

    out['nearest_pocket'] = recall_at_k(y_test, nearest_pocket(x_train, y_train, x_test))
    log('  nearest pocket    %s' % out['nearest_pocket'])

    out['learned'] = recall_at_k(y_test, learned(x_train, y_train, x_test, log=log))
    log('  learned           %s' % out['learned'])

    # The claim is only that the pocket helps if a pocket-aware method beats the
    # pocket-blind prior. Stated explicitly so a wash cannot be written up as a
    # success.
    prior = out['frequency_prior'].get('recall@10') or 0
    best = max((out['nearest_pocket'].get('recall@10') or 0),
               (out['learned'].get('recall@10') or 0))
    out['_verdict'] = {
        'prior_recall@10': prior, 'best_pocket_aware_recall@10': best,
        'delta': round(best - prior, 4),
        'uses_the_pocket': bool(best - prior >= 0.02),
    }
    log('  verdict: pocket-aware beats prior by %+.4f at recall@10'
        % out['_verdict']['delta'])
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--regimes', nargs='+', default=list(REGIMES))
    ap.add_argument('--scheme', default='brics', choices=('brics', 'scaffold', 'both'))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    results = {r: run(args.release, r, args.scheme) for r in args.regimes}
    path = args.out or os.path.join(args.release, 'results', 'reverse.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    print('wrote', path)
