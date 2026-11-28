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
    """A protein-side descriptor, as independent of the ligand as this corpus allows.

    Every field must be derivable from the pocket rather than from the bound
    complex. Anything computed around the ligand encodes the ligand's size and
    would make the no-pocket ablation look healthy while the model quietly
    cheats.

    **The raw residue count used to be the first field and has been removed.**
    The pocket is defined as residues within 6.5 A of a ligand heavy atom, so
    that count correlates with ligand heavy-atom count at r = 0.64 across 25,811
    complexes — a volume measured around the ligand, which is precisely what the
    paragraph above forbids. It mattered because the per-atom ligand-only arm
    carries no global size feature, so the count handed the pocket arm something
    about the ligand that its control could not see.

    Measured before removing it, on `unseen_protein`: ligand only 0.6246, pocket
    with every feature 0.6782, pocket without the count 0.6678, and the count on
    its own 0.6098 — *below* the ligand-only arm. So the effect was never the
    leak. Dropping the count costs 0.0104 macro AUC and leaves a +0.0432
    ablation delta that rests on composition rather than on size.

    What remains are fractions, which are invariant to how many residues line
    the site. They are still computed over a ligand-defined neighbourhood, so
    they are not fully ligand-independent: scoring a genuinely apo protein needs
    a pocket finder to define the site first, and that distribution shift is
    untested. See the README.
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
    example = {
        'key': (record['pdb_id'], record['comp_id']),
        'smiles': record['smiles'],
        'uniprot': record['uniprot'],
        'inchikey': record.get('inchikey'),
        'atom_features': atoms,
        'pocket_features': np.array(pocket_features(record), dtype=np.float32),
        'labels': label_matrix(record, n_heavy),
        'n_heavy': n_heavy,
    }
    features, mask = neighbour_features(record)
    if features is not None and features.shape[0] == n_heavy:
        # A mismatch means the stored coordinates belong to a different atom
        # ordering than the SMILES this example is built from, which would pair
        # every atom with someone else's neighbourhood. Dropping the geometry
        # is the safe answer; a silent misalignment is not.
        example['neighbour_features'] = features
        example['neighbour_mask'] = mask
    return example


def load_examples(release, split_map=None, subset=None, geometry=True):
    """Examples from the corpus, optionally restricted to one split subset.

    Geometry lives in two places and is merged here. Complexes built after the
    corpus started storing coordinates carry them in the record; earlier ones
    are filled from the backfill sidecar, which is written separately because
    the build appends to the record shards continuously and rewriting a file
    another process is appending to is how corpora get silently corrupted.
    """
    sidecar = {}
    if geometry:
        try:
            from pharmacophore.backfill_geometry import load_sidecar, key_of
            sidecar = load_sidecar(release)
        except Exception:
            sidecar = {}
    out = []
    for record in iter_records(release):
        if record.get('rejected'):
            continue
        key = (record['pdb_id'], record['comp_id'])
        if split_map is not None:
            where = split_map.get(key)
            if where is None or (subset and where != subset):
                continue
        if sidecar and not record.get('pocket_atoms'):
            extra = sidecar.get(key_of(record))
            if extra:
                record = dict(record, ligand_xyz=extra['ligand_xyz'],
                              pocket_atoms=extra['pocket_atoms'])
        example = record_to_example(record)
        if example is not None:
            out.append(example)
    return out


def geometry_coverage(examples):
    """How many examples actually carry a neighbourhood.

    Reported rather than assumed: training the geometric arm on a corpus that
    is only half backfilled would quietly compare a pocket-aware model on half
    the data against a ligand-only model on all of it.
    """
    with_geometry = sum(1 for e in examples if e.get('neighbour_features') is not None)
    return {'examples': len(examples), 'with_geometry': with_geometry,
            'fraction': round(with_geometry / max(len(examples), 1), 4)}


# --------------------------------------------------------------------------
# Geometric pocket neighbourhoods
# --------------------------------------------------------------------------

# Neighbours kept per ligand atom. Sixteen covers the first shell comfortably --
# a buried atom has roughly 10-14 protein atoms within 5 A -- without padding
# most rows with far-away atoms that carry no signal.
N_NEIGHBOURS = 16

# Gaussian centres for the distance expansion, in Angstroms. A raw distance is a
# poor input: the difference between 2.8 and 3.2 A decides whether something is
# a hydrogen bond, while 7 and 8 A are equally irrelevant. Expanding onto
# overlapping Gaussians lets the model weight those regimes separately.
RBF_CENTRES = np.array([2.0, 2.75, 3.5, 4.25, 5.0, 5.75, 6.5, 8.0], dtype=np.float32)
RBF_WIDTH = 0.75

NEIGHBOUR_ELEMENTS = ['C', 'N', 'O', 'S']

HYDROPHOBIC_RES = {'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO', 'PHE', 'TRP'}
AROMATIC_RES = {'PHE', 'TYR', 'TRP', 'HIS'}
CATIONIC_RES = {'LYS', 'ARG', 'HIS'}
ANIONIC_RES = {'ASP', 'GLU'}
POLAR_RES = {'SER', 'THR', 'ASN', 'GLN', 'CYS', 'TYR'}
BACKBONE_ATOMS = {'N', 'CA', 'C', 'O', 'OXT'}

# distance RBFs + element one-hot (+other) + 5 residue classes + backbone flag
NEIGHBOUR_DIM = len(RBF_CENTRES) + len(NEIGHBOUR_ELEMENTS) + 1 + 5 + 1


def neighbour_features(record, k=N_NEIGHBOURS):
    """(n_heavy, k, NEIGHBOUR_DIM) and a (n_heavy, k) mask of real neighbours.

    For each ligand heavy atom, its `k` nearest pocket atoms described by how
    far away they are and what they are. This is what the composition fractions
    could never express: a fraction says the pocket is 11% anionic, this says
    *this* atom has a carboxylate oxygen 2.8 A away.

    Only distances are used, never raw coordinates, so the representation is
    invariant to rotation and translation by construction rather than by
    augmentation — there is nothing for the model to learn about the arbitrary
    frame a PDB file happens to be deposited in.

    Returns `None` when the record has no stored geometry, which is how a
    corpus that is only partly backfilled stays usable.
    """
    ligand_xyz = record.get('ligand_xyz')
    pocket_atoms = record.get('pocket_atoms')
    if not ligand_xyz or not pocket_atoms:
        return None, None

    lig = np.asarray(ligand_xyz, dtype=np.float32)
    coords = np.asarray([[a[3], a[4], a[5]] for a in pocket_atoms], dtype=np.float32)
    n_heavy = lig.shape[0]
    features = np.zeros((n_heavy, k, NEIGHBOUR_DIM), dtype=np.float32)
    mask = np.zeros((n_heavy, k), dtype=bool)
    if not coords.size:
        return features, mask

    # Describe each pocket atom once; the same atom is a neighbour of many
    # ligand atoms and rebuilding this inside the loop is what made an early
    # version unusably slow.
    identity = np.zeros((len(pocket_atoms), NEIGHBOUR_DIM - len(RBF_CENTRES)),
                        dtype=np.float32)
    for i, atom in enumerate(pocket_atoms):
        element, resname, atom_name = atom[0], atom[1], atom[2]
        offset = 0
        if element in NEIGHBOUR_ELEMENTS:
            identity[i, NEIGHBOUR_ELEMENTS.index(element)] = 1.0
        else:
            identity[i, len(NEIGHBOUR_ELEMENTS)] = 1.0
        offset = len(NEIGHBOUR_ELEMENTS) + 1
        for j, group in enumerate((HYDROPHOBIC_RES, AROMATIC_RES, CATIONIC_RES,
                                   ANIONIC_RES, POLAR_RES)):
            if resname in group:
                identity[i, offset + j] = 1.0
        if atom_name in BACKBONE_ATOMS:
            identity[i, offset + 5] = 1.0

    distances = np.linalg.norm(lig[:, None, :] - coords[None, :, :], axis=2)
    take = min(k, coords.shape[0])
    nearest = np.argpartition(distances, take - 1, axis=1)[:, :take]
    for atom_index in range(n_heavy):
        order = nearest[atom_index][np.argsort(distances[atom_index][nearest[atom_index]])]
        d = distances[atom_index][order]
        rbf = np.exp(-((d[:, None] - RBF_CENTRES[None, :]) ** 2) / (2 * RBF_WIDTH ** 2))
        features[atom_index, :take, :len(RBF_CENTRES)] = rbf
        features[atom_index, :take, len(RBF_CENTRES):] = identity[order]
        mask[atom_index, :take] = True
    return features, mask


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
