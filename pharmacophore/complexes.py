"""Choosing which protein–ligand complexes enter the corpus, and why each one did not.

The selection rules are the ones `redock.py` already uses on the docking page —
the same exclusion list, the same covalent test, the same completeness check
against the chemical-component definition — because a ligand the platform
refuses to redock is not a ligand it should learn pharmacophores from either.

Every rejection is recorded with its reason. A corpus whose exclusions are
invisible cannot be audited, and the counts are part of the release manifest:
"how many structures did we lose, and to what" is a question a reader of the
paper will ask.
"""
import os
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redock  # noqa: E402  - the platform's own ligand detection, already tested
from pharmacophore import sources  # noqa: E402

RDLogger.DisableLog('rdApp.*')

# Deliberately lower than the docking panel's drug-like floor of 12. A corpus
# built to learn which *fragments* bind a pocket cannot exclude fragments:
# crystallographic fragment screens deposit ligands of 7-15 heavy atoms, and
# 3PTB's benzamidine - the canonical validated case on this platform - has 9.
# Solvents, buffers and ions are already removed by name in redock.py, so the
# floor only has to exclude the small junk that list misses. Heavy-atom count is
# recorded on every record, so a drug-like-only subset stays one filter away.
MIN_LIGAND_HEAVY_ATOMS = 8
MAX_LIGAND_HEAVY_ATOMS = 100     # above this it is a peptide or a cofactor assembly
MAX_RESOLUTION = 2.5
POCKET_RADIUS = 6.5              # Å from any ligand heavy atom

# Cofactors and prosthetic groups. These are genuine ligands with genuine
# contacts, but they appear in thousands of unrelated structures and would
# dominate any frequency-based signal — a model that learns "haem binds
# everywhere" has learned about the corpus, not about chemistry. Excluded from
# the ligand role; they remain part of the receptor environment.
COFACTORS = {
    'HEM', 'HEC', 'HEA', 'HEB', 'NAD', 'NAI', 'NAP', 'NDP', 'FAD', 'FMN', 'ADP', 'ATP',
    'AMP', 'GDP', 'GTP', 'GNP', 'ANP', 'ACP', 'SAM', 'SAH', 'COA', 'ACO', 'TPP', 'PLP',
    'BCL', 'CLA', 'BPH', 'PHO', 'F43', 'B12', 'COB', 'MQ7', 'UQ1', 'PQQ', 'MGD',
}

# Crystallisation additives beyond the list `redock.py` carries. The first pilot
# kept 10CX on the strength of 2PE — a polyethylene glycol oligomer lying in a
# surface groove, which is antifreeze rather than a ligand. A hand-maintained
# blocklist will always have holes like that, which is why the ring test below
# does the real work and this list only catches the common names quickly.
EXTRA_ARTEFACTS = {
    # polyethylene glycols and their oligomers
    '2PE', 'PGE', 'PG0', 'PG5', 'PG6', '7PE', '12P', '15P', '33O', 'XPE', 'M2M', 'DIO',
    'PE3', 'PE4', 'PE5', 'PE8', 'P33', 'P4C', 'P6G', '1PG', '2PG', 'SPM', 'SPD',
    # polyols, alcohols and small solvents
    'MRD', 'BU3', 'BU1', 'PDO', 'PGO', 'PGR', 'ETX', 'IPA', 'MOH', 'EOH', 'ACN', 'ACE',
    'TFA', 'DMF', 'NMO', 'URE', 'GBL', 'SIN', 'OXL', 'MLA', 'MLI', 'SUC', 'TAR',
    # buffers and salts
    'EPE', 'MES', 'TRS', 'BTB', 'BIS', 'CAC', 'PIN', 'MPO', 'NHE', 'TAM', 'B3P',
    'AZI', 'THJ', 'WO4', 'MOO', 'VO4', 'BEF', 'ALF', 'PER', 'BR', 'FLC', 'CIT',
    # detergents and lipids used for crystallisation
    'LDA', 'LMT', 'BOG', 'OTG', 'C8E', 'HTG', 'HTO', 'UND', 'OCT', 'DAO', 'D10', 'D12',
    'MYR', 'PLM', 'STE', 'OLA', 'OLB', 'OLC', 'PEE', 'PCW', 'LHG', 'CDL', 'DGA',
}


def is_artefact(comp_id, mol):
    """Is this a crystallisation additive rather than something that was bound?

    Two tests, cheap first. The name list catches the usual suspects. The ring
    test does the rest: cryoprotectants and buffers — PEG, glycerol, ethylene
    glycol, MPD, Tris — are flexible acyclic chains, while essentially every
    drug and fragment carries at least one ring. Benzamidine, indinavir and
    donepezil all pass; 2PE does not.

    **This excludes genuinely acyclic ligands too** — some peptidomimetics,
    fatty-acid ligands and acyclic inhibitors — and that bias is recorded in the
    manifest rather than hidden. Trading a minority of real acyclic ligands for
    the removal of thousands of PEG molecules is the right side of that bargain,
    and the rejection log makes the cost visible.
    """
    if comp_id in EXTRA_ARTEFACTS:
        return 'known_artefact'
    from rdkit.Chem import rdMolDescriptors
    if rdMolDescriptors.CalcNumRings(mol) == 0:
        return 'acyclic_likely_additive'
    return None


class Rejected(Exception):
    """Carries the reason a complex did not make it in."""

    def __init__(self, reason, detail=''):
        super().__init__(f'{reason}: {detail}' if detail else reason)
        self.reason = reason
        self.detail = detail


def entry_is_usable(meta):
    """Resolution and method gate, from the entry metadata rather than the file."""
    info = meta.get('rcsb_entry_info', {})
    methods = info.get('experimental_method') or ''
    if 'X-ray' not in methods and 'X-RAY' not in str(methods).upper():
        raise Rejected('not_xray', str(methods)[:40])
    res = info.get('resolution_combined')
    res = res[0] if isinstance(res, list) and res else res
    if res is None:
        raise Rejected('no_resolution')
    if res > MAX_RESOLUTION:
        raise Rejected('resolution_too_low', f'{res} A')
    return float(res)


def ligand_candidates(pdb_text):
    """Drug-like, non-covalent, single-component ligands in this structure.

    Reuses `redock.ligand_candidates`, which already groups bonded HETATM
    residues into one entity (so a ligand deposited as several linked components
    is not mistaken for several small ligands) and flags covalent attachment by
    heavy-atom distance.
    """
    candidates, parsed = redock.ligand_candidates(pdb_text)
    keep = []
    for c in candidates:
        if c['comp_id'] in COFACTORS:
            continue
        if len(c['components']) > 1:
            continue                       # multi-component: cannot rebuild reliably
        if c['covalent_contacts']:
            continue                       # covalent: not comparable with the rest
        if not (MIN_LIGAND_HEAVY_ATOMS <= c['heavy_atoms'] <= MAX_LIGAND_HEAVY_ATOMS):
            continue
        keep.append(c)
    return keep, parsed


def ligand_mol(candidate, comp_id):
    """The crystal ligand as an RDKit molecule with correct bond orders.

    `redock.crystal_mol` maps the deposited coordinates onto the chemical
    component definition, which is also the completeness check: a ligand missing
    heavy atoms will not match its own definition and is rejected rather than
    quietly entering the corpus truncated.
    """
    try:
        template = redock.fetch_component(comp_id)
    except redock.RedockError as e:
        raise Rejected('no_component_definition', str(e)[:60])
    try:
        return redock.crystal_mol(candidate['atoms'], template)
    except redock.RedockError as e:
        raise Rejected('ligand_incomplete_or_unmatched', str(e)[:60])


def pocket_residues(pdb_text, ligand_atoms, radius=POCKET_RADIUS):
    """Protein residues with any heavy atom within `radius` of the ligand.

    Returns residue keys and their atoms. This is the receptor side of the
    complex and the input to the pocket descriptors; the ligand itself is not
    part of it.
    """
    lig_xyz = np.array([a['xyz'] for a in ligand_atoms])
    residues = {}
    for line in pdb_text.splitlines():
        if line.startswith('ENDMDL'):
            break
        if not line.startswith('ATOM'):
            continue
        alt = line[16:17]
        if alt not in (' ', 'A'):
            continue
        element = (line[76:78].strip() or line[12:16].strip()[:1]).upper()
        if element in ('H', 'D'):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        key = (line[17:20].strip(), line[21:22].strip(), line[22:27].strip())
        residues.setdefault(key, []).append({'name': line[12:16].strip(), 'el': element,
                                             'xyz': xyz, 'line': line})
    if not residues:
        raise Rejected('no_protein_atoms')

    near = {}
    for key, atoms in residues.items():
        xyz = np.array([a['xyz'] for a in atoms])
        d = np.linalg.norm(lig_xyz[:, None, :] - xyz[None, :, :], axis=2)
        if d.min() <= radius:
            near[key] = atoms
    if not near:
        raise Rejected('ligand_not_in_a_pocket')
    return near


def uniprot_for_chain(pdb_id, meta, chain):
    """The UniProt accession for the chain the ligand sits against.

    Without it the complex cannot be joined to bioactivity or clustered by
    protein for the split, so a complex that cannot be mapped is rejected.
    """
    entity_ids = (meta.get('rcsb_entry_container_identifiers', {})
                  .get('polymer_entity_ids') or [])
    for entity_id in entity_ids:
        try:
            entity = sources.polymer_entity(pdb_id, entity_id)
        except Exception:
            continue
        ids = entity.get('rcsb_polymer_entity_container_identifiers', {})
        chains = (ids.get('auth_asym_ids') or []) + (ids.get('asym_ids') or [])
        if chain and chain not in chains:
            continue
        for ref in (ids.get('reference_sequence_identifiers') or []):
            if ref.get('database_name') == 'UniProt':
                return ref.get('database_accession'), entity_id
    raise Rejected('no_uniprot_mapping', f'chain {chain}')


def select(pdb_id):
    """Every qualifying complex in one PDB entry.

    One entry can contribute several complexes — the same ligand in several
    chains, or different ligands in different pockets. They are emitted
    separately and deduplicated later by (uniprot, ligand InChIKey), because two
    copies of the same complex in one crystal are one observation, not two.
    """
    meta = sources.entry_metadata(pdb_id)
    resolution = entry_is_usable(meta)
    try:
        pdb_text = sources.structure_pdb(pdb_id)
    except FileNotFoundError:
        raise Rejected('no_pdb_format', 'mmCIF-only entry')

    candidates, _parsed = ligand_candidates(pdb_text)
    if not candidates:
        raise Rejected('no_drug_like_ligand')

    out = []
    for cand in candidates:
        try:
            mol = ligand_mol(cand, cand['comp_id'])
            artefact = is_artefact(cand['comp_id'], mol)
            if artefact:
                raise Rejected(artefact, cand['comp_id'])
            uniprot, entity_id = uniprot_for_chain(pdb_id, meta, cand['chain'])
            pocket = pocket_residues(pdb_text, cand['atoms'])
        except Rejected as e:
            out.append({'pdb_id': pdb_id, 'comp_id': cand['comp_id'],
                        'rejected': e.reason, 'detail': e.detail})
            continue
        smiles = Chem.MolToSmiles(mol)
        out.append({
            'pdb_id': pdb_id,
            'resolution': resolution,
            'comp_id': cand['comp_id'],
            'chain': cand['chain'],
            'resi': cand['resi'],
            'uniprot': uniprot,
            'entity_id': entity_id,
            'smiles': smiles,
            'inchikey': Chem.MolToInchiKey(mol),
            'heavy_atoms': cand['heavy_atoms'],
            'ligand_atoms': cand['atoms'],
            'ligand_mol': mol,
            'pocket': pocket,
            'pdb_text': pdb_text,
            'rejected': None,
        })
    return out


def summarise_rejections(records):
    return dict(Counter(r['rejected'] for r in records if r.get('rejected')))
