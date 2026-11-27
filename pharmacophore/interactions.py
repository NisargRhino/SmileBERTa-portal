"""Per-ligand-atom interaction fingerprints — the labels the models learn from.

This is where a crystal structure becomes supervision. For every heavy atom of
the bound ligand we record which typed interactions it makes with the protein:
hydrogen bonds, salt bridges, hydrophobic contacts, π-stacking, halogen bonds,
metal coordination. That set, with its geometry, is what a medicinal chemist
means by a pharmacophore, and unlike a docking score it is an observation.

Three details decide whether the labels are right, and all three were wrong in
the first working version of this file:

1. **Hydrogens.** Crystal structures do not have them. Without them the
   detectors see no donors at all. They are added at pH 7.4 by `protonate.py`,
   which keeps the depositor's residue numbering.
2. **Formal charges.** RDKit perceives a deprotonated carboxylate from
   coordinates as a neutral oxygen, so trypsin's Asp189 reads as an acid and its
   salt bridge to benzamidine is typed as an ordinary hydrogen bond. Charges are
   therefore assigned here from residue and atom names, which is deterministic
   and needs no tool.
3. **Ionisation of the ligand.** Benzamidine deposited as a neutral amidine is
   an amidinium at physiological pH. Protonating the ligand is what makes its
   charge, and therefore its salt bridge, visible.

Detection itself is ProLIF's, not ours: its SMARTS patterns and geometric
criteria are published and maintained, and hand-rolled geometry is how a corpus
acquires quiet systematic errors.
"""
import os
import tempfile

import numpy as np

from pharmacophore import protonate as P

# Interactions kept, in a fixed order so a feature vector means the same thing
# in every record and in every model that consumes one.
INTERACTIONS = [
    'Hydrophobic', 'HBDonor', 'HBAcceptor', 'Cationic', 'Anionic',
    'PiStacking', 'PiCation', 'CationPi', 'XBDonor', 'XBAcceptor',
    'MetalDonor', 'MetalAcceptor',
]

# Formal charges at pH 7.4, by residue and atom name. Only the side chains that
# are reliably ionised are listed: histidine is deliberately left neutral, since
# its pKa sits near physiological pH and guessing it wrong in either direction
# would plant a systematic error in the labels.
PROTEIN_CHARGES = {
    ('ASP', 'OD2'): -1, ('GLU', 'OE2'): -1,
    ('LYS', 'NZ'): +1, ('ARG', 'NH2'): +1,
}
# C-terminal carboxylate and N-terminal ammonium are handled separately, by
# position rather than by name.


def assign_protein_charges(mol):
    """Set formal charges on an RDKit protein from residue and atom names.

    Returns the number of atoms charged, for the manifest: a structure where
    nothing was charged usually means the PDB residue metadata did not survive,
    and that is worth noticing rather than silently producing uncharged labels.
    """
    n = 0
    for atom in mol.GetAtoms():
        info = atom.GetMonomerInfo()
        if info is None:
            continue
        key = (info.GetResidueName().strip().upper(), info.GetName().strip().upper())
        charge = PROTEIN_CHARGES.get(key)
        if charge is None:
            continue
        atom.SetFormalCharge(charge)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
        n += 1
    return n


def protein_molecule(pdb_text, protonate=True):
    """A ProLIF Molecule for the protein, protonated and charged.

    MDAnalysis is used for the PDB parsing because ProLIF is built on it, and
    because it preserves author residue numbering — which `protonate.py` goes to
    some trouble to keep.
    """
    import MDAnalysis as mda
    import prolif as plf

    text, _added = P.protonated_pdb(pdb_text) if protonate else (pdb_text, 0)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'protein.pdb')
        with open(path, 'w') as f:
            f.write(text)
        universe = mda.Universe(path)
        selection = universe.select_atoms('protein')
        if not len(selection):
            raise ValueError('no protein atoms after protonation')
        mol = plf.Molecule.from_mda(selection, NoImplicit=False)
    assign_protein_charges(mol)
    return mol


def ligand_molecule(rdkit_mol, protonate=True):
    import prolif as plf
    mol = P.protonated_mol(rdkit_mol) if protonate else rdkit_mol
    return plf.Molecule.from_rdkit(mol), mol


def fingerprint(ligand_mol, protein_pdb_text, interactions=None, protonate=True):
    """Per-atom interactions for one complex.

    Returns a dict with:
      `per_atom`   — heavy-atom index -> {interaction type -> [partner residues]}
      `contacts`   — the flat list, each with type, residue, distance, ligand atoms
      `by_type`    — counts per interaction type
      `n_heavy`    — heavy atom count, so a vector can be built without the mol

    Ligand atom indices are reported against the **heavy-atom** molecule passed
    in, not the protonated one, so they line up with the SMILES stored in the
    corpus and with whatever a model tokenises later.
    """
    import prolif as plf

    names = interactions or INTERACTIONS
    protein = protein_molecule(protein_pdb_text, protonate=protonate)
    lig_plf, protonated = ligand_molecule(ligand_mol, protonate=protonate)

    fp = plf.Fingerprint(names)
    fp.run_from_iterable([lig_plf], protein, progress=False)

    heavy_index = _heavy_atom_map(protonated, ligand_mol)
    per_atom, contacts, by_type = {}, [], {}
    for (_lres, pres), data in (fp.ifp[0] if fp.ifp else {}).items():
        residue = str(pres)
        for name, metas in data.items():
            for meta in (metas if isinstance(metas, (list, tuple)) else [metas]):
                indices = (meta.get('parent_indices') or {}).get('ligand') or ()
                mapped = sorted({heavy_index[i] for i in indices if i in heavy_index})
                distance = float(meta.get('distance') or 0.0)
                contacts.append({'type': name, 'residue': residue, 'distance': round(distance, 2),
                                 'ligand_atoms': mapped})
                by_type[name] = by_type.get(name, 0) + 1
                for idx in mapped:
                    per_atom.setdefault(idx, {}).setdefault(name, []).append(residue)
    return {'per_atom': per_atom, 'contacts': contacts, 'by_type': by_type,
            'n_heavy': ligand_mol.GetNumHeavyAtoms()}


def _heavy_atom_map(protonated, original):
    """Map atom indices in the protonated ligand back to the heavy-atom molecule.

    Open Babel appends hydrogens after the existing atoms and leaves heavy atoms
    in order, so the mapping is positional — but it is verified by element here
    rather than assumed, because a silent off-by-one would mislabel every atom
    in the corpus.
    """
    heavy_elements = [a.GetAtomicNum() for a in original.GetAtoms()]
    mapping, seen = {}, 0
    for i, atom in enumerate(protonated.GetAtoms()):
        if atom.GetAtomicNum() == 1:
            continue
        if seen < len(heavy_elements) and atom.GetAtomicNum() == heavy_elements[seen]:
            mapping[i] = seen
            seen += 1
        else:
            return {i: i for i in range(len(heavy_elements))}   # orders diverged
    return mapping


def feature_vector(result, interactions=None):
    """A per-atom matrix of shape (n_heavy, n_interaction_types), 0/1."""
    names = interactions or INTERACTIONS
    matrix = np.zeros((result['n_heavy'], len(names)), dtype=np.int8)
    for idx, types in result['per_atom'].items():
        if idx >= result['n_heavy']:
            continue
        for name in types:
            if name in names:
                matrix[idx, names.index(name)] = 1
    return matrix


def pocket_feature_profile(result, interactions=None):
    """What the pocket demanded, independent of which ligand satisfied it.

    This is the target side of the same alphabet: the counts of each interaction
    type the pocket formed. It is what direction 2 predicts from a protein alone
    and what a fragment's own profile is matched against.
    """
    names = interactions or INTERACTIONS
    return {name: result['by_type'].get(name, 0) for name in names}
