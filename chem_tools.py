"""RDKit-powered chemistry utilities used by the builder, analysis and docking endpoints."""
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, QED, rdMolDescriptors, rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point3D

MAX_BUILDER_ATOMS = 400
MAX_CONFORMERS = 30


class ChemError(ValueError):
    pass


# ── Parsing ──────────────────────────────────────────────────────────────────

def mol_from_smiles(smiles):
    if not smiles or not isinstance(smiles, str):
        raise ChemError('SMILES string is required')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ChemError('Could not parse SMILES: ' + smiles[:80])
    return mol


def embed_3d(mol, seed=0xC0FFEE, optimize=True):
    """Add hydrogens, embed with ETKDG and relax with MMFF94 (UFF fallback)."""
    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(molh, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(molh, params) != 0:
            raise ChemError('3D embedding failed for this molecule')
    if optimize:
        if AllChem.MMFFHasAllMoleculeParams(molh):
            AllChem.MMFFOptimizeMolecule(molh, maxIters=2000)
        else:
            AllChem.UFFOptimizeMolecule(molh, maxIters=2000)
    return molh


# ── Properties ───────────────────────────────────────────────────────────────

def analyze(mol):
    """Full medicinal-chemistry profile for a molecule."""
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    lipinski_violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    try:
        qed = QED.qed(mol)
    except Exception:
        qed = None
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = None
    return {
        'smiles': Chem.MolToSmiles(mol),
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'inchikey': inchikey,
        'molecular_weight': round(mw, 2),
        'exact_mass': round(Descriptors.ExactMolWt(mol), 4),
        'log_p': round(logp, 2),
        'molar_refractivity': round(Crippen.MolMR(mol), 2),
        'hydrogen_bond_donors': hbd,
        'hydrogen_bond_acceptors': hba,
        'tpsa': round(tpsa, 2),
        'rotatable_bonds': rotb,
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'rings': rdMolDescriptors.CalcNumRings(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'fraction_sp3': round(rdMolDescriptors.CalcFractionCSP3(mol), 3),
        'formal_charge': Chem.GetFormalCharge(mol),
        'stereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        'qed': round(qed, 3) if qed is not None else None,
        'lipinski_violations': lipinski_violations,
        'lipinski_pass': lipinski_violations <= 1,
        'veber_pass': rotb <= 10 and tpsa <= 140,
    }


def depict_svg(mol, width=420, height=320, highlight=None, dark=False):
    mol = Chem.Mol(mol)
    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    if dark:
        try:
            rdMolDraw2D.SetDarkMode(opts)
        except Exception:
            pass
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.08
    drawer.DrawMolecule(mol, highlightAtoms=highlight or [])
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


# ── Builder ──────────────────────────────────────────────────────────────────

def mol_from_atoms(atoms, bonds):
    """Build an RDKit molecule (with explicit H, no implicit H) from builder JSON."""
    if not atoms:
        raise ChemError('The molecule is empty')
    if len(atoms) > MAX_BUILDER_ATOMS:
        raise ChemError(f'Too many atoms (max {MAX_BUILDER_ATOMS})')
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))
    for i, a in enumerate(atoms):
        try:
            atom = Chem.Atom(str(a.get('el', 'C')))
        except Exception:
            raise ChemError(f'Unknown element "{a.get("el")}"')
        atom.SetFormalCharge(int(a.get('charge', 0) or 0))
        atom.SetNoImplicit(True)
        rw.AddAtom(atom)
        conf.SetAtomPosition(i, Point3D(float(a.get('x', 0)), float(a.get('y', 0)), float(a.get('z', 0))))
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE,
                 1.5: Chem.BondType.AROMATIC}
    for b in bonds or []:
        i, j = int(b['a']), int(b['b'])
        if i == j or not (0 <= i < len(atoms)) or not (0 <= j < len(atoms)):
            raise ChemError('Bond references an invalid atom')
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, order_map.get(b.get('order', 1), Chem.BondType.SINGLE))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(mol)
    except Chem.AtomValenceException as e:
        raise ChemError('Valence error: ' + str(e).split('\n')[0])
    except Exception as e:
        raise ChemError('Could not sanitize molecule: ' + str(e).split('\n')[0])
    return mol


def mol_to_builder_json(mol, conf_id=-1):
    conf = mol.GetConformer(conf_id)
    atoms = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        atoms.append({'el': atom.GetSymbol(), 'x': round(p.x, 4), 'y': round(p.y, 4), 'z': round(p.z, 4),
                      'charge': atom.GetFormalCharge()})
    kek = Chem.Mol(mol)
    try:
        Chem.Kekulize(kek, clearAromaticFlags=True)
    except Exception:
        kek = mol
    bonds = []
    for bond in kek.GetBonds():
        bt = bond.GetBondType()
        order = 2 if bt == Chem.BondType.DOUBLE else 3 if bt == Chem.BondType.TRIPLE else 1
        bonds.append({'a': bond.GetBeginAtomIdx(), 'b': bond.GetEndAtomIdx(), 'order': order})
    return {'atoms': atoms, 'bonds': bonds}


def _forcefield(mol, name, conf_id=-1):
    name = (name or 'MMFF94').upper()
    if name in ('MMFF94', 'MMFF94S'):
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s' if name == 'MMFF94S' else 'MMFF94')
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id, nonBondedThresh=100.0)
            if ff is not None:
                return ff, ('MMFF94s' if name == 'MMFF94S' else 'MMFF94')
        name = 'UFF'  # fall back when MMFF lacks parameters (metals, exotic atoms)
    if not AllChem.UFFHasAllMoleculeParams(mol):
        raise ChemError('No force-field parameters available for this molecule')
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id, vdwThresh=100.0)
    return ff, 'UFF'


def optimize_geometry(mol, forcefield='MMFF94', max_steps=2000, n_frames=30):
    """Minimise the conformer, recording intermediate frames so the client can animate it."""
    ff, used = _forcefield(mol, forcefield)
    ff.Initialize()
    e0 = ff.CalcEnergy()
    n = mol.GetNumAtoms()
    frames = []
    chunk = 8
    steps = 0
    converged = False
    while steps < max_steps:
        status = ff.Minimize(maxIts=chunk)
        steps += chunk
        frames.append((round(ff.CalcEnergy(), 4), ff.Positions()))
        if status == 0:
            converged = True
            break
    # Subsample evenly so the client gets at most n_frames snapshots to animate.
    n_frames = max(2, min(int(n_frames), 120))
    if len(frames) > n_frames:
        idx = sorted({round(i * (len(frames) - 1) / (n_frames - 1)) for i in range(n_frames)})
        frames = [frames[i] for i in idx]
    frames = [{'energy': e, 'coords': [round(v, 3) for v in pos]} for e, pos in frames]
    pos = ff.Positions()
    conf = mol.GetConformer()
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]))
    return {
        'forcefield': used,
        'energy_initial': round(e0, 4),
        'energy_final': round(ff.CalcEnergy(), 4),
        'energy_unit': 'kcal/mol',
        'steps': steps,
        'converged': converged,
        'frames': frames,
    }


def conformer_search(mol, n_confs=10, forcefield='MMFF94', seed=42):
    n_confs = max(1, min(int(n_confs), MAX_CONFORMERS))
    molh = Chem.AddHs(Chem.RemoveHs(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    cids = list(AllChem.EmbedMultipleConfs(molh, numConfs=n_confs * 3, params=params))
    if not cids:
        raise ChemError('Conformer embedding failed')
    used = 'UFF'
    results = []
    if forcefield.upper().startswith('MMFF') and AllChem.MMFFHasAllMoleculeParams(molh):
        out = AllChem.MMFFOptimizeMoleculeConfs(molh, maxIters=2000)
        used = 'MMFF94'
    else:
        out = AllChem.UFFOptimizeMoleculeConfs(molh, maxIters=2000)
    for cid, (_, energy) in zip(cids, out):
        results.append((energy, cid))
    results.sort()
    # Prune near-duplicates after optimisation.
    kept = []
    heavy = Chem.RemoveHs(molh)
    for energy, cid in results:
        dup = False
        for _, kcid in kept:
            rms = AllChem.GetConformerRMS(molh, cid, kcid)
            if rms < 0.25:
                dup = True
                break
        if not dup:
            kept.append((energy, cid))
        if len(kept) >= n_confs:
            break
    e_min = kept[0][0]
    confs = []
    for energy, cid in kept:
        data = mol_to_builder_json(molh, cid)
        confs.append({'energy': round(energy, 4), 'relative': round(energy - e_min, 4),
                      'atoms': data['atoms'], 'bonds': data['bonds']})
    del heavy
    return {'forcefield': used, 'conformers': confs}


def builder_export(mol, fmt):
    fmt = fmt.lower()
    if fmt in ('mol', 'sdf'):
        return Chem.MolToMolBlock(mol) + ('$$$$\n' if fmt == 'sdf' else '')
    if fmt == 'pdb':
        return Chem.MolToPDBBlock(mol)
    if fmt == 'xyz':
        return Chem.MolToXYZBlock(mol)
    if fmt == 'smiles':
        return Chem.MolToSmiles(Chem.RemoveHs(mol))
    raise ChemError('Unsupported format: ' + fmt)
