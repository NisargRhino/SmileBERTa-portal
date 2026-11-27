"""Turn corpus records into model inputs and per-atom labels.

The task is node classification: for each heavy atom of a ligand, which typed
interactions does it make with the protein? Labels come straight from the
corpus, which read them off crystal geometry.

Two properties of this data decide how everything downstream must be built, and
both are measured rather than assumed:

* **The labels are sparse.** Around 18% of ligand atoms carry any interaction at
  all, and the rarer types are far below that. A model predicting "nothing" for
  every atom scores about 82% accuracy, which is why accuracy is not reported
  anywhere and per-type precision, recall and AUC are.
* **One ligand usually appears once.** A model can reproduce the training labels
  by memorising molecules without ever using the protein, which is what the
  no-pocket ablation exists to detect.
"""
import json
import os
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

from pharmacophore.build import iter_records
from pharmacophore.interactions import INTERACTIONS

RDLogger.DisableLog('rdApp.*')

# Atom features that need no protein and no learned representation — the inputs
# for the fingerprint baselines the spec requires before any network.
ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Se']
HYBRIDISATIONS = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                  Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                  Chem.HybridizationType.SP3D2]


def _onehot(value, options):
    vector = [0] * (len(options) + 1)
    vector[options.index(value) if value in options else len(options)] = 1
    return vector


def atom_features(mol, idx):
    """Per-atom descriptors: element, hybridisation, charge, ring and H-bond role.

    Deliberately simple and interpretable. If a random forest on these plus a
    pocket descriptor matches a transformer, that is the finding, and it is the
    same finding the classification benchmark already produced once.
    """
    atom = mol.GetAtomWithIdx(idx)
    return (
        _onehot(atom.GetSymbol(), ELEMENTS)
        + _onehot(atom.GetHybridization(), HYBRIDISATIONS)
        + [
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetTotalNumHs(),
            atom.GetDegree(),
            float(atom.GetSymbol() in ('N', 'O', 'S')),          # possible acceptor
            float(atom.GetTotalNumHs() > 0 and atom.GetSymbol() in ('N', 'O')),  # donor
        ]
    )


def pocket_features(record):
    """A protein-side descriptor computable without the ligand.

    Every field here must be derivable from the pocket alone. Anything computed
    from the bound complex — pocket volume measured around the ligand, for
    instance — encodes the ligand's size and would make the no-pocket ablation
    look healthy while the model quietly cheats.
    """
    residues = [r.split()[0] for r in record.get('pocket_residues', []) if r.split()]
    counts = Counter(residues)
    total = max(len(residues), 1)
    hydrophobic = sum(counts[r] for r in ('ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE',
                                          'TRP', 'PRO', 'GLY'))
    aromatic = sum(counts[r] for r in ('PHE', 'TYR', 'TRP', 'HIS'))
    positive = sum(counts[r] for r in ('LYS', 'ARG', 'HIS'))
    negative = sum(counts[r] for r in ('ASP', 'GLU'))
    polar = sum(counts[r] for r in ('SER', 'THR', 'ASN', 'GLN', 'TYR', 'CYS'))
    return [
        len(residues),
        hydrophobic / total, aromatic / total, positive / total,
        negative / total, polar / total,
        (positive - negative) / total,
    ]


def label_matrix(record, n_heavy):
    """(n_heavy, n_interaction_types) binary labels from the corpus record."""
    matrix = np.zeros((n_heavy, len(INTERACTIONS)), dtype=np.int8)
    for idx_str, types in (record.get('per_atom') or {}).items():
        idx = int(idx_str)
        if idx >= n_heavy:
            continue
        for name in types:
            if name in INTERACTIONS:
                matrix[idx, INTERACTIONS.index(name)] = 1
    return matrix


def record_to_example(record):
    """One corpus record -> per-atom features, pocket features and labels."""
    mol = Chem.MolFromSmiles(record['smiles'])
    if mol is None:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy == 0 or n_heavy != mol.GetNumAtoms():
        mol = Chem.RemoveHs(mol)
        n_heavy = mol.GetNumAtoms()
    atoms = np.array([atom_features(mol, i) for i in range(n_heavy)], dtype=np.float32)
    return {
        'key': (record['pdb_id'], record['comp_id']),
        'smiles': record['smiles'],
        'uniprot': record['uniprot'],
        'inchikey': record.get('inchikey'),
        'atom_features': atoms,
        'pocket_features': np.array(pocket_features(record), dtype=np.float32),
        'labels': label_matrix(record, n_heavy),
        'n_heavy': n_heavy,
    }


def load_examples(release, split_map=None, subset=None):
    """Examples from the corpus, optionally restricted to one split subset."""
    out = []
    for record in iter_records(release):
        if record.get('rejected'):
            continue
        key = (record['pdb_id'], record['comp_id'])
        if split_map is not None:
            where = split_map.get(key)
            if where is None or (subset and where != subset):
                continue
        example = record_to_example(record)
        if example is not None:
            out.append(example)
    return out


def promiscuous_ligands(release):
    """InChIKeys seen bound to more than one distinct protein.

    These are the only complexes where the pocket is actually in contention.
    For a ligand that appears once, memorising the molecule is the correct
    answer and a ligand-only model loses nothing by ignoring the protein --
    and since only about one ligand in eight is promiscuous, a corpus-wide
    ablation average is mostly measuring cases that cannot distinguish the two
    arms. Restricting to this set is what makes the ablation sensitive.
    """
    seen = {}
    for record in iter_records(release):
        if record.get('rejected') or not record.get('inchikey'):
            continue
        seen.setdefault(record['inchikey'], set()).add(record.get('uniprot'))
    return {key for key, proteins in seen.items() if len(proteins) > 1}


def atom_mask(examples, inchikeys):
    """Per-atom boolean mask over `flatten`/`predict` output order."""
    flags = []
    for example in examples:
        promiscuous = example.get('inchikey') in inchikeys
        flags.extend([promiscuous] * example['n_heavy'])
    return np.array(flags, dtype=bool)


def flatten(examples, with_pocket=True):
    """Stack per-atom rows into (X, Y, groups) for the fingerprint baselines.

    `with_pocket=False` produces the ablation input: the same atoms with no
    protein information at all. If performance barely moves between the two,
    the model is not using the protein and the pocket-conditioned half of the
    project does not work, whatever the headline number says.
    """
    xs, ys, groups = [], [], []
    for n, example in enumerate(examples):
        atoms = example['atom_features']
        if with_pocket:
            pocket = np.tile(example['pocket_features'], (atoms.shape[0], 1))
            atoms = np.concatenate([atoms, pocket], axis=1)
        xs.append(atoms)
        ys.append(example['labels'])
        groups.extend([n] * atoms.shape[0])
    if not xs:
        return np.zeros((0, 0)), np.zeros((0, len(INTERACTIONS))), np.array([])
    return np.vstack(xs), np.vstack(ys), np.array(groups)


def morgan_atom_environment(smiles, radius=2, n_bits=256):
    """Per-atom circular-environment bits, for a stronger fingerprint baseline."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    rows = []
    for i in range(mol.GetNumAtoms()):
        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, i) if radius else []
        submol = Chem.PathToSubmol(mol, env) if env else None
        target = submol if submol is not None and submol.GetNumAtoms() else mol
        fp = AllChem.GetMorganFingerprintAsBitVect(target, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.array(rows, dtype=np.float32)


def label_statistics(examples):
    """Per-type positive rates — the base rates every metric is read against."""
    if not examples:
        return {}
    labels = np.vstack([e['labels'] for e in examples])
    total_atoms = labels.shape[0]
    return {
        'atoms': int(total_atoms),
        'complexes': len(examples),
        'any_interaction_rate': round(float((labels.sum(axis=1) > 0).mean()), 4),
        'per_type': {name: {'positives': int(labels[:, i].sum()),
                            'rate': round(float(labels[:, i].mean()), 5)}
                     for i, name in enumerate(INTERACTIONS)},
    }


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    args = ap.parse_args()
    examples = load_examples(args.release)
    print(json.dumps(label_statistics(examples), indent=2))
