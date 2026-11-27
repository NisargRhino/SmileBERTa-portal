"""Datasets for the honest evaluation, loaded exactly once and the same way everywhere.

Two datasets, both small enough to hold in memory:

* **Classification** — `drug_classification_data_df.csv`: 8,336 compounds, 12
  therapeutic categories, space-separated in the `tags` column. 1,350 compounds
  carry more than one category, so this is a multi-label problem that the shipped
  model answers with a single label string.
* **Fragment prediction** — `ml-project/drug_fragments.csv`: BRICS decompositions
  produced by `ml-project/FBDD-discovery.py`. One molecule yields **several**
  fragments, which is the fact every fragment metric has to respect: a prediction
  is correct if it matches any of that molecule's fragments, not just the first.

Everything is canonicalised through RDKit on load, so a molecule has one spelling
no matter which file it came from, and a compound whose SMILES RDKit cannot parse
is dropped and counted rather than silently carried into a split.
"""
import csv
import os
from collections import Counter, OrderedDict

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFICATION_CSV = os.path.join(REPO, 'drug_classification_data_df.csv')
FRAGMENT_CSV = os.path.join(REPO, 'ml-project', 'drug_fragments.csv')
LABEL_OPTIONS_CSV = os.path.join(REPO, 'drugclassoptions_final1.csv')

# The 12 therapeutic categories, fixed in this order so every table, confusion
# matrix and split file in the repo lists them identically.
CATEGORIES = [
    'antiinfective', 'antineoplastic', 'cns', 'cardio', 'antiinflammatory',
    'gastrointestinal', 'hematologic', 'respiratorysystem', 'dermatologic',
    'reproductivecontrol', 'lipidregulating', 'urological',
]


def canonical(smiles):
    """RDKit canonical SMILES, or None when the molecule will not parse."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def scaffold_of(smiles, include_chirality=False):
    """Bemis–Murcko scaffold SMILES.

    Molecules with no ring system (and therefore no Murcko scaffold) come back as
    `''`. They are *not* all one group: lumping every acyclic compound together
    would create a single huge group that has to land in one split, so callers
    give each of them its own group — see `splits.group_by_scaffold`.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except Exception:
        return None


def load_classification():
    """[{smiles, labels, scaffold}], canonicalised, with the drop report."""
    rows, dropped, seen = [], Counter(), {}
    with open(CLASSIFICATION_CSV) as f:
        for raw in csv.DictReader(f):
            smi = canonical(raw['smiles'])
            if smi is None:
                dropped['unparseable'] += 1
                continue
            labels = [t for t in raw['tags'].split() if t]
            unknown = [t for t in labels if t not in CATEGORIES]
            if unknown or not labels:
                dropped['unknown_or_missing_label'] += 1
                continue
            if smi in seen:
                # Same molecule twice: union the labels rather than keeping a
                # duplicate that could straddle two splits.
                dropped['duplicate_smiles'] += 1
                merged = seen[smi]
                merged['labels'] = sorted(set(merged['labels']) | set(labels),
                                          key=CATEGORIES.index)
                continue
            rec = {'smiles': smi, 'labels': sorted(set(labels), key=CATEGORIES.index),
                   'scaffold': scaffold_of(smi), 'source_smiles': raw['smiles']}
            seen[smi] = rec
            rows.append(rec)
    return rows, dict(dropped)


def load_fragments():
    """[{smiles, fragments, scaffold}] — one record per molecule, all its BRICS fragments.

    The source file has one row per (molecule, fragment) pair. Grouping them here
    is what lets `exact match` mean "matched one of this molecule's fragments"
    instead of "matched whichever row happened to come first".
    """
    by_mol, dropped = OrderedDict(), Counter()
    with open(FRAGMENT_CSV) as f:
        for raw in csv.DictReader(f):
            parent = canonical(raw['Original_SMILES'])
            if parent is None:
                dropped['unparseable_parent'] += 1
                continue
            frag = (raw['Fragment'] or '').strip()
            if not frag or Chem.MolFromSmiles(frag) is None:
                dropped['unparseable_fragment'] += 1
                continue
            by_mol.setdefault(parent, {'smiles': parent, 'fragments': [],
                                       'scaffold': scaffold_of(parent)})
            if frag not in by_mol[parent]['fragments']:
                by_mol[parent]['fragments'].append(frag)
    return list(by_mol.values()), dict(dropped)


def label_options():
    with open(LABEL_OPTIONS_CSV) as f:
        return [r['name'].strip() for r in csv.DictReader(f) if r['name'].strip()]


def strip_dummies(frag_smiles):
    """A BRICS fragment with its `[n*]` attachment points removed.

    BRICS marks cut bonds with dummy atoms, which are not part of the parent
    molecule, so they have to go before a fragment can be tested as a
    substructure of it.
    """
    mol = Chem.MolFromSmiles(frag_smiles)
    if mol is None:
        return None
    edit = Chem.RWMol(mol)
    for idx in sorted((a.GetIdx() for a in edit.GetAtoms() if a.GetAtomicNum() == 0), reverse=True):
        edit.RemoveAtom(idx)
    out = edit.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out if out.GetNumAtoms() else None


def is_substructure(frag_smiles, parent_smiles):
    """Is this fragment genuinely part of that molecule?

    The spec calls this out on its own line, and rightly: the shipped pipeline
    can answer with a fragment from a *different* molecule when the model's raw
    output does not parse and the nearest-reference path fires. That is a
    hallucination, and it is only visible if something checks.
    """
    frag = strip_dummies(frag_smiles)
    parent = Chem.MolFromSmiles(parent_smiles) if parent_smiles else None
    if frag is None or parent is None:
        return False
    try:
        return parent.HasSubstructMatch(frag, useChirality=False)
    except Exception:
        return False


def summarise():
    """Numbers for the write-up's dataset section."""
    cls, cls_dropped = load_classification()
    frags, frag_dropped = load_fragments()
    per_label = Counter(l for r in cls for l in r['labels'])
    per_count = Counter(len(r['labels']) for r in cls)
    return {
        'classification': {
            'compounds': len(cls),
            'dropped': cls_dropped,
            'per_label': {c: per_label[c] for c in CATEGORIES},
            'labels_per_compound': dict(sorted(per_count.items())),
            'multi_label_compounds': sum(v for k, v in per_count.items() if k > 1),
            'distinct_scaffolds': len({r['scaffold'] for r in cls}),
            'acyclic_compounds': sum(1 for r in cls if r['scaffold'] == ''),
        },
        'fragments': {
            'molecules': len(frags),
            'fragment_pairs': sum(len(r['fragments']) for r in frags),
            'dropped': frag_dropped,
            'fragments_per_molecule': round(
                sum(len(r['fragments']) for r in frags) / max(len(frags), 1), 2),
            'distinct_scaffolds': len({r['scaffold'] for r in frags}),
        },
    }


if __name__ == '__main__':
    import json
    print(json.dumps(summarise(), indent=2))
