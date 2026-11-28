"""Serve the pharmacophore model: a molecule in, its binding atoms out.

Separate from `train.py` on purpose. The evaluation runs hold a test set back
and must never see it; this loads a checkpoint and answers questions. Keeping
them apart is what stops a convenience import from quietly giving an evaluation
access to weights trained on its own test split.

Two modes, and the difference matters to anyone reading the output:

**With a pocket.** Requires a pose — the model scores each atom from its
distances to nearby protein atoms — so this needs a complex, not just a SMILES
string and a protein name. Given one, the prediction is specific to that site.

**Without a pocket.** Falls back to the ligand-only arm, which answers "which
atoms of this molecule tend to make interactions" rather than "which atoms bind
*this* protein". That is a weaker and different claim, and the response says so
rather than letting a caller assume otherwise.
"""
import os

import numpy as np

CHECKPOINT = os.environ.get('SB_PHARMACOPHORE_MODEL',
                            os.path.join('corpus', 'v1', 'models', 'geometric.pt'))

_cache = {}


def available(path=None):
    return os.path.exists(path or CHECKPOINT)


def _model(path=None):
    """Load once per process. The model is ~8 MB, the API is memory-bound."""
    path = path or CHECKPOINT
    if path not in _cache:
        from pharmacophore import model as Model
        _cache[path] = Model.load_checkpoint(path)
    return _cache[path]


def predict(smiles, pocket_record=None, path=None, threshold=0.5):
    """Per-atom interaction probabilities for one molecule.

    `pocket_record` is a corpus-shaped dict carrying `ligand_xyz` and
    `pocket_atoms`; without it the pocket is masked and the answer is
    ligand-only.
    """
    import torch
    from rdkit import Chem
    from pharmacophore import dataset as D
    from pharmacophore import model as Model
    from pharmacophore.interactions import INTERACTIONS

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('could not parse SMILES')
    mol = Chem.RemoveHs(mol)
    n_heavy = mol.GetNumAtoms()
    if n_heavy == 0:
        raise ValueError('molecule has no heavy atoms')

    net, blob = _model(path)
    atoms = np.array([D.atom_features(mol, i) for i in range(n_heavy)],
                     dtype=np.float32)

    neighbours = np.zeros((n_heavy, D.N_NEIGHBOURS, D.NEIGHBOUR_DIM), dtype=np.float32)
    nmask = np.zeros((n_heavy, D.N_NEIGHBOURS), dtype=bool)
    used_pocket = False
    if pocket_record:
        features, mask = D.neighbour_features(pocket_record)
        if features is not None and features.shape[0] == n_heavy:
            neighbours, nmask = features, mask
            used_pocket = True

    with torch.no_grad():
        logits = net(torch.from_numpy(atoms).unsqueeze(0),
                     torch.from_numpy(neighbours).unsqueeze(0),
                     torch.from_numpy(nmask).unsqueeze(0),
                     torch.ones(1, n_heavy, dtype=torch.bool))
        probabilities = torch.sigmoid(logits)[0].numpy()

    per_atom = []
    for i in range(n_heavy):
        types = {name: round(float(probabilities[i][j]), 4)
                 for j, name in enumerate(INTERACTIONS)
                 if probabilities[i][j] >= threshold}
        per_atom.append({
            'atom': i,
            'element': mol.GetAtomWithIdx(i).GetSymbol(),
            'score': round(float(probabilities[i].max()), 4),
            'interactions': types,
        })

    order = np.argsort(-probabilities.max(axis=1))
    return {
        'smiles': Chem.MolToSmiles(mol),
        'n_heavy_atoms': n_heavy,
        'used_pocket': used_pocket,
        # Stated in the payload, not just the docs: a caller that forgets to
        # send a pocket should not be able to present the answer as
        # protein-specific.
        'interpretation': ('per-atom interactions predicted for this pocket'
                           if used_pocket else
                           'ligand-only: which atoms of this molecule tend to '
                           'interact, not which bind a particular protein'),
        'per_atom': per_atom,
        'pharmacophore_atoms': [int(i) for i in order
                                if probabilities[i].max() >= threshold],
        'model': {
            'kind': blob.get('kind'),
            'trained_on': (blob.get('metadata') or {}).get('complexes'),
            'test_macro_auc': (blob.get('metadata') or {}).get('test_macro_auc'),
            'ablation_delta': (blob.get('metadata') or {}).get('ablation_delta'),
        },
    }


def pocket_from_pdb(pdb_text, ligand_mol):
    """Build the pocket record the model needs from a structure and a bound pose."""
    from pharmacophore import complexes as C
    pocket = C.pocket_residues(pdb_text, [
        {'xyz': tuple(ligand_mol.GetConformer().GetAtomPosition(i))}
        for i in range(ligand_mol.GetNumAtoms())])
    ligand_xyz, pocket_atoms = C.pocket_geometry(pocket, ligand_mol)
    return {'ligand_xyz': ligand_xyz, 'pocket_atoms': pocket_atoms}
