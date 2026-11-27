"""Add hydrogens at pH 7.4 without losing the crystallographer's residue numbering.

Interaction typing needs hydrogens and formal charges: without them benzamidine
is a neutral amidine rather than an amidinium, trypsin's Asp189 is a neutral
acid rather than a carboxylate, and the salt bridge between them — the most
cited interaction in the complex — simply is not detected.

Open Babel will add those hydrogens at a given pH, and the docking pipeline
already uses it for exactly this. What it will not do is preserve residue
numbering: it renumbers sequentially on output, which turned Asp189 into
Asp171 in testing. For trypsin, numbered by the chymotrypsin convention with
gaps, that is a shift of eighteen residues, and every downstream join —
UniProt mapping, protein clustering, the residue names shown to a user — would
silently refer to the wrong thing.

So the hydrogens come from Open Babel and the bookkeeping does not: heavy atoms
keep their original ATOM records verbatim, and each added hydrogen inherits the
residue of the heavy atom it is bonded to.
"""
import numpy as np
from openbabel import openbabel as ob
from openbabel import pybel

ob.obErrorLog.SetOutputLevel(0)

PH = 7.4


def _atom_line(serial, name, resname, chain, resseq, icode, xyz, element):
    return ('ATOM  %5d %-4s %3s %1s%4s%1s   %8.3f%8.3f%8.3f  1.00  0.00          %2s'
            % (serial % 100000, name[:4], resname[:3], chain or ' ', str(resseq)[:4],
               icode or ' ', xyz[0], xyz[1], xyz[2], element.rjust(2)[:2]))


def parse_atoms(pdb_text, records=('ATOM',)):
    """Heavy-atom ATOM records of the first model, first altloc."""
    out = []
    for line in pdb_text.splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith(records):
            continue
        if line[16:17] not in (' ', 'A'):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if element in ('H', 'D'):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        out.append({'name': line[12:16].strip(), 'resname': line[17:20].strip(),
                    'chain': line[21:22].strip(), 'resseq': line[22:26].strip(),
                    'icode': line[26:27].strip(), 'xyz': xyz, 'element': element,
                    'line': line})
    return out


def protonated_pdb(pdb_text, ph=PH, records=('ATOM',)):
    """`pdb_text` with polar hydrogens added at `ph`, numbering intact.

    Returns (pdb_text_with_h, n_added). Heavy atoms are emitted from their
    original records, so anything downstream that reads residue identity sees
    exactly what the depositor wrote.
    """
    heavy = parse_atoms(pdb_text, records)
    if not heavy:
        return pdb_text, 0

    stripped = '\n'.join(a['line'] for a in heavy) + '\nEND\n'
    mol = pybel.readstring('pdb', stripped)
    before = mol.OBMol.NumAtoms()
    # (polaronly=False, correctForPH=True, pH): full hydrogens so that aromatic
    # and aliphatic donors are typed correctly; the geometry checks in the
    # interaction detectors need the hydrogen positions, not just their count.
    mol.OBMol.AddHydrogens(False, True, ph)

    coords, elements = [], []
    for atom in ob.OBMolAtomIter(mol.OBMol):
        coords.append((atom.GetX(), atom.GetY(), atom.GetZ()))
        elements.append(ob.GetSymbol(atom.GetAtomicNum()).upper())

    if before != len(heavy):
        # Open Babel dropped or merged something; fall back rather than risk
        # mapping hydrogens onto the wrong residues.
        return pdb_text, 0

    heavy_xyz = np.array([a['xyz'] for a in heavy])
    # Hydrogens are grouped under the heavy atom they sit on, then emitted
    # immediately after it. Writing them all at the end instead splits every
    # residue across two parts of the file, and MDAnalysis then reads the
    # residue as fragmented and ProLIF refuses it outright — which is how a
    # homodimer like 1HSG fails while a monomer passes.
    attached = {}
    added = 0
    for i in range(before, len(coords)):
        if elements[i] != 'H':
            continue
        d = np.linalg.norm(heavy_xyz - np.array(coords[i]), axis=1)
        parent = int(np.argmin(d))
        attached.setdefault(parent, []).append(coords[i])
        added += 1

    lines, serial = [], 0
    for idx, atom in enumerate(heavy):
        serial += 1
        lines.append(atom['line'])
        for n, xyz in enumerate(attached.get(idx, []), 1):
            serial += 1
            name = ('H%s%d' % (atom['name'][1:3], n))[:4] or 'H'
            lines.append(_atom_line(serial, name, atom['resname'], atom['chain'],
                                    atom['resseq'], atom['icode'], xyz, 'H'))
    return '\n'.join(lines) + '\nEND\n', added


def protonated_mol(rdkit_mol, ph=PH):
    """An RDKit ligand protonated at `ph`, so ionisable groups carry their charge.

    Benzamidine's amidine becomes an amidinium here, which is what makes its
    salt bridge detectable.
    """
    from rdkit import Chem
    obmol = pybel.readstring('mol', Chem.MolToMolBlock(rdkit_mol))
    obmol.OBMol.AddHydrogens(False, True, ph)
    out = Chem.MolFromMolBlock(obmol.write('mol'), removeHs=False)
    return out if out is not None else rdkit_mol
