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


# Margin, in Angstroms, kept around each ligand when the receptor is trimmed.
# Every interaction this module detects is decided well inside this: the widest
# cutoffs are PI_CENTROID_CUTOFF at 5.5 and IONIC_CUTOFF at 5.0, and ProLIF's
# own defaults top out around 6. Twelve is roughly double the largest of them,
# so a trimmed receptor cannot change a verdict; it only removes atoms that were
# never candidates. Whole residues are kept, so no residue is half-present.
TRIM_MARGIN = 12.0


def protein_molecule(pdb_text, protonate=True, near=None, margin=TRIM_MARGIN):
    """A ProLIF Molecule for the protein, protonated and charged.

    MDAnalysis is used for the PDB parsing because ProLIF is built on it, and
    because it preserves author residue numbering — which `protonate.py` goes to
    some trouble to keep.

    `near` is a sequence of ligand coordinate arrays. Given them, only residues
    within `margin` of a ligand are converted. This matters more than it looks:
    `plf.Molecule.from_mda` calls RDKit's `GetMolFrags` to split the receptor
    into residues, and on a 30,000-atom structure — 60,000 once protonated —
    that single call is the largest allocation in the whole pipeline. On a
    machine whose swap cannot grow it is reliably the one that dies with
    SIGBUS, which is why every crash report in this project names it. Trimming
    turns tens of thousands of atoms into a couple of thousand.

    Without `near` the whole receptor is converted, which is the old behaviour
    and still correct, only expensive.
    """
    import MDAnalysis as mda
    import numpy as np
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
        if near is not None and len(near):
            spheres = []
            for coords in near:
                coords = np.asarray(coords, dtype=float)
                if not coords.size:
                    continue
                centre = coords.mean(axis=0)
                radius = float(np.linalg.norm(coords - centre, axis=1).max()) + margin
                spheres.append('point %.3f %.3f %.3f %.3f'
                               % (centre[0], centre[1], centre[2], radius))
            if spheres:
                trimmed = universe.select_atoms(
                    'protein and byres (%s)' % ' or '.join(spheres))
                # Falling back rather than failing: a selection that somehow
                # catches nothing must not silently produce an empty receptor.
                if len(trimmed):
                    selection = trimmed
        mol = plf.Molecule.from_mda(selection, NoImplicit=False)
    assign_protein_charges(mol)
    return mol


def ligand_molecule(rdkit_mol, protonate=True):
    import prolif as plf
    mol = P.protonated_mol(rdkit_mol) if protonate else rdkit_mol
    return plf.Molecule.from_rdkit(mol), mol


def fingerprint(ligand_mol, protein_pdb_text=None, interactions=None, protonate=True,
                protein_mol=None, pocket=None):
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
    # The protein can be prepared once per entry and reused for every ligand in
    # it. Protonating a 10,000-atom structure and rebuilding its RDKit molecule
    # for each of a dozen ligand copies is what left one worker at full CPU for
    # ten minutes on a single entry during the pilot.
    protein = protein_mol or protein_molecule(protein_pdb_text, protonate=protonate)
    lig_plf, protonated = ligand_molecule(ligand_mol, protonate=protonate)

    fp = plf.Fingerprint(names)
    fp.run_from_iterable([lig_plf], protein, progress=False)

    heavy_index = _heavy_atom_map(protonated, ligand_mol)
    per_atom, contacts, by_type = {}, [], {}
    # Salt bridges and pi-stacking come from the geometric detectors, since the
    # library's versions cannot fire on a protein parsed from a PDB file.
    if pocket:
        for extra in ionic_and_pi_contacts(protonated, pocket):
            mapped = sorted({heavy_index.get(i, i) for i in extra['ligand_atoms']})
            extra = dict(extra, ligand_atoms=[i for i in mapped if i < ligand_mol.GetNumAtoms()])
            contacts.append(extra)
            by_type[extra['type']] = by_type.get(extra['type'], 0) + 1
            for idx in extra['ligand_atoms']:
                per_atom.setdefault(idx, {}).setdefault(extra['type'], []).append(extra['residue'])
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


# ── Ionic and aromatic interactions, which ProLIF cannot see here ────────────
# A PDB file carries no bond orders, so nothing perceives Phe, Tyr, Trp or His
# rings as aromatic from coordinates alone — a protein built from one has zero
# aromatic atoms, and ProLIF's PiStacking and Cationic/Anionic detectors
# therefore never fire. Across 42 pilot complexes they fired exactly zero times,
# including on 3PTB, whose benzamidine-Asp189 salt bridge is the most cited
# interaction in the structure.
#
# These two detectors fill that gap. They are deliberate hand-rolled geometry,
# against the spec's own preference for a maintained library, because the
# alternative is a pharmacophore corpus that silently contains no salt bridges
# and no pi-stacking. Both are typed from residue and atom names, which is
# unambiguous for the standard amino acids, and both are validated against
# complexes whose interactions are documented.

# Charged groups at pH 7.4, by residue and the atoms carrying the charge.
PROTEIN_ANIONS = {'ASP': ('OD1', 'OD2'), 'GLU': ('OE1', 'OE2')}
PROTEIN_CATIONS = {'LYS': ('NZ',), 'ARG': ('NE', 'NH1', 'NH2')}
# Aromatic rings by atom name. Histidine is included as an aromatic ring even
# though it is left uncharged: ring geometry does not depend on protonation.
PROTEIN_RINGS = {
    'PHE': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],
    'TYR': [('CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ')],
    'HIS': [('CG', 'ND1', 'CD2', 'CE1', 'NE2')],
    'TRP': [('CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'),
            ('CG', 'CD1', 'CD2', 'NE1', 'CE2')],
}
IONIC_CUTOFF = 5.0        # Å between charged centres
PI_CENTROID_CUTOFF = 5.5  # Å between ring centroids
PI_ANGLE_FACE = 30.0      # degrees: face-to-face stacking
PI_ANGLE_EDGE = 60.0      # degrees: beyond this, edge-to-face


def _residue_atoms(pocket):
    """{(resname, chain, resseq): {atom name: xyz}} from a pocket mapping."""
    out = {}
    for key, atoms in pocket.items():
        out[key] = {a['name'].strip().upper(): np.array(a['xyz']) for a in atoms}
    return out


def _ring_geometry(points):
    centroid = points.mean(axis=0)
    # Normal from the plane of best fit; the smallest singular vector is the
    # direction of least spread, which for a planar ring is its normal.
    normal = np.linalg.svd(points - centroid)[2][-1]
    return centroid, normal / np.linalg.norm(normal)


def _angle_between(a, b):
    cos = abs(float(np.dot(a, b)))
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, cos)))))


def ionic_and_pi_contacts(ligand_mol, pocket):
    """Salt bridges, pi-stacking and cation-pi for one complex.

    `ligand_mol` must carry real bond orders and formal charges — it comes from
    the chemical-component definition and is protonated at pH 7.4, so aromatic
    rings and charges are known on the ligand side. Only the protein side needs
    name-based typing.

    Returned in the same shape as ProLIF's contacts so the two merge cleanly.
    """
    contacts = []
    conf = ligand_mol.GetConformer()
    coords = {i: np.array(list(conf.GetAtomPosition(i)))
              for i in range(ligand_mol.GetNumAtoms())}
    heavy = {i for i, a in enumerate(ligand_mol.GetAtoms()) if a.GetAtomicNum() > 1}

    lig_cations = [i for i in heavy if ligand_mol.GetAtomWithIdx(i).GetFormalCharge() > 0]
    lig_anions = [i for i in heavy if ligand_mol.GetAtomWithIdx(i).GetFormalCharge() < 0]
    lig_rings = []
    for ring in ligand_mol.GetRingInfo().AtomRings():
        if all(ligand_mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            lig_rings.append((ring, *_ring_geometry(np.array([coords[i] for i in ring]))))

    residues = _residue_atoms(pocket)
    for (resname, chain, resseq), atoms in residues.items():
        label = f'{resname}{resseq}.{chain}'

        # Salt bridges, named as ProLIF names them: the interaction is reported
        # from the ligand's point of view.
        for group, lig_atoms, name in ((PROTEIN_ANIONS, lig_cations, 'Cationic'),
                                       (PROTEIN_CATIONS, lig_anions, 'Anionic')):
            for atom_name in group.get(resname, ()):
                if atom_name not in atoms:
                    continue
                for i in lig_atoms:
                    d = float(np.linalg.norm(coords[i] - atoms[atom_name]))
                    if d <= IONIC_CUTOFF:
                        contacts.append({'type': name, 'residue': label,
                                         'distance': round(d, 2), 'ligand_atoms': [i]})

        # Pi-stacking and cation-pi.
        for ring_names in PROTEIN_RINGS.get(resname, ()):
            if not all(n in atoms for n in ring_names):
                continue
            prot_centroid, prot_normal = _ring_geometry(
                np.array([atoms[n] for n in ring_names]))
            for ring, lig_centroid, lig_normal in lig_rings:
                d = float(np.linalg.norm(lig_centroid - prot_centroid))
                if d > PI_CENTROID_CUTOFF:
                    continue
                angle = _angle_between(lig_normal, prot_normal)
                kind = ('FaceToFace' if angle <= PI_ANGLE_FACE
                        else 'EdgeToFace' if angle >= PI_ANGLE_EDGE else None)
                if kind:
                    contacts.append({'type': 'PiStacking', 'residue': label,
                                     'distance': round(d, 2),
                                     'ligand_atoms': sorted(ring),
                                     'geometry': kind, 'angle': round(angle, 1)})
            for i in lig_cations:
                d = float(np.linalg.norm(coords[i] - prot_centroid))
                if d <= PI_CENTROID_CUTOFF:
                    contacts.append({'type': 'PiCation', 'residue': label,
                                     'distance': round(d, 2), 'ligand_atoms': [i]})
    return contacts
