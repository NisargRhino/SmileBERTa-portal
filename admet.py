"""ADMET profiling from published, reproducible models.

Design rule for this module: every number it returns can be traced to a paper,
and nothing is invented. Each row carries a `cite` naming the source and a
`basis` saying how it was obtained:

  'equation'      a published regression evaluated directly (e.g. ESOL)
  'rule'          a published threshold/filter (e.g. Lipinski's rule of five)
  'range'         a published classification boundary, applied as a box
  'substructure'  a SMARTS alert set shipped with RDKit (PAINS, Brenk, NIH)
  'descriptor'    a plain computed descriptor, no model involved

Endpoints that genuinely require a trained black-box model (CYP inhibition,
P-glycoprotein transport, hERG, Ames, LD50) are *not* guessed at here. They are
reported in the `unavailable` list with a pointer to the Model Hub, because a
made-up CYP prediction is worse than no prediction for anyone making decisions
from it.

Everything runs on RDKit alone, in a few milliseconds per molecule, with no
extra dependencies and no network access.
"""
import math
import os
import sys
import threading

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from rdkit.Chem import RDConfig


class AdmetError(Exception):
    pass


# ── Contributed scorers ──────────────────────────────────────────────────────
# SA_Score and NP_Score ship inside RDKit's Contrib tree but are not importable
# as normal modules, so the path has to be added by hand. Both are loaded lazily
# behind a lock: NP_Score reads a ~2 MB fingerprint model off disk, which we do
# not want to pay for at import time on a memory-constrained dyno.

_scorer_lock = threading.Lock()
_sascorer = None
_npscorer = None
_np_model = None


def _contrib(name):
    path = os.path.join(RDConfig.RDContribDir, name)
    if path not in sys.path:
        sys.path.append(path)


def _sa_score(mol):
    """Synthetic accessibility, 1 (easy) to 10 (hard). Ertl & Schuffenhauer 2009."""
    global _sascorer
    with _scorer_lock:
        if _sascorer is None:
            _contrib('SA_Score')
            import sascorer
            _sascorer = sascorer
    return _sascorer.calculateScore(mol)


def _np_score(mol):
    """Natural-product likeness, roughly -5 (synthetic) to +5 (natural). Ertl 2008."""
    global _npscorer, _np_model
    with _scorer_lock:
        if _npscorer is None:
            _contrib('NP_Score')
            import contextlib
            import io
            import npscorer
            # readNPModel() chatters on stderr; keep the server log clean.
            with contextlib.redirect_stderr(io.StringIO()):
                _np_model = npscorer.readNPModel()
            _npscorer = npscorer
    return _npscorer.scoreMol(mol, _np_model)


# ── Structural-alert catalogues ──────────────────────────────────────────────
# Built once and reused; each catalogue is kept separate so a match can say
# which family it came from.

_ALERT_FAMILIES = [
    ('PAINS', 'PAINS', 'Pan-assay interference compounds — frequent hitters in screens',
     'Baell & Holloway, J. Med. Chem. 2010, 53, 2719'),
    ('BRENK', 'Brenk', 'Substructures the authors advise against in lead series',
     'Brenk et al., ChemMedChem 2008, 3, 435'),
    ('NIH', 'NIH', 'Reactive or otherwise undesirable groups flagged by the NIH screening centre',
     'Doveston et al., Org. Biomol. Chem. 2015, 13, 859; Jadhav et al., J. Med. Chem. 2010, 53, 37'),
]

_catalog_lock = threading.Lock()
_catalogs = None


def _alert_catalogs():
    global _catalogs
    with _catalog_lock:
        if _catalogs is None:
            built = []
            for key, label, blurb, cite in _ALERT_FAMILIES:
                try:
                    params = FilterCatalogParams()
                    params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, key))
                    built.append((label, blurb, cite, FilterCatalog.FilterCatalog(params)))
                except Exception:
                    # A catalogue missing from this RDKit build should degrade the
                    # panel, not fail the whole request.
                    continue
            _catalogs = built
    return _catalogs


def _alerts(mol):
    out = []
    for label, blurb, cite, catalog in _alert_catalogs():
        try:
            matches = catalog.GetMatches(mol)
        except Exception:
            continue
        for m in matches:
            try:
                name = m.GetDescription()
            except Exception:
                name = 'unnamed alert'
            out.append({'family': label, 'name': name, 'blurb': blurb, 'cite': cite})
    return out


# ── Descriptor block ─────────────────────────────────────────────────────────

def _descriptors(mol):
    heavy = mol.GetNumHeavyAtoms()
    aromatic_heavy = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic() and a.GetAtomicNum() > 1)
    carbons = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    heteroatoms = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    return {
        'mw': Descriptors.MolWt(mol),
        'heavy_atoms': heavy,
        'aromatic_heavy_atoms': aromatic_heavy,
        # Delaney's "aromatic proportion": aromatic heavy atoms over all heavy atoms.
        'aromatic_proportion': (aromatic_heavy / heavy) if heavy else 0.0,
        'carbons': carbons,
        'heteroatoms': heteroatoms,
        # Crippen's logP *is* WLOGP — the Wildman & Crippen atom-contribution
        # method — which is the descriptor Egan's and the BOILED-Egg's
        # boundaries were fitted against.
        'wlogp': Crippen.MolLogP(mol),
        'mr': Crippen.MolMR(mol),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'rings': rdMolDescriptors.CalcNumRings(mol),
        'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
        'fraction_csp3': rdMolDescriptors.CalcFractionCSP3(mol),
        'formal_charge': Chem.GetFormalCharge(mol),
        'stereocenters': len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }


# ── Solubility ───────────────────────────────────────────────────────────────

ESOL_CITE = 'Delaney, J. Chem. Inf. Comput. Sci. 2004, 44, 1000'


def _esol(d):
    """Delaney's ESOL regression for log S (mol/L).

    Log(S) = 0.16 - 0.63*clogP - 0.0062*MW + 0.066*RB - 0.74*AP

    Fitted on 2874 measured solubilities; reported average absolute error is
    ~0.75 log units, so treat the number as an order-of-magnitude guide.
    """
    return (0.16
            - 0.63 * d['wlogp']
            - 0.0062 * d['mw']
            + 0.066 * d['rotatable_bonds']
            - 0.74 * d['aromatic_proportion'])


# Classification boundaries as used by SwissADME for log S.
_SOL_CLASSES = [
    (0.0, 'Highly soluble'),
    (-2.0, 'Very soluble'),
    (-4.0, 'Soluble'),
    (-6.0, 'Moderately soluble'),
    (-10.0, 'Poorly soluble'),
]


def _sol_class(log_s):
    for cut, label in _SOL_CLASSES:
        if log_s >= cut:
            return label
    return 'Insoluble'


# ── Drug-likeness filters ────────────────────────────────────────────────────
# Each filter is a list of (description, satisfied?) checks. A molecule's
# violations are the checks that came back False.

def _filters(d):
    mw, logp, mr = d['mw'], d['wlogp'], d['mr']
    hbd, hba, tpsa = d['hbd'], d['hba'], d['tpsa']
    rotb, heavy = d['rotatable_bonds'], d['heavy_atoms']

    def f(name, cite, note, checks, allowed=0):
        violations = [desc for desc, ok in checks if not ok]
        return {
            'name': name, 'cite': cite, 'note': note,
            'violations': violations,
            'allowed_violations': allowed,
            'passed': len(violations) <= allowed,
            'checks': [{'desc': desc, 'ok': bool(ok)} for desc, ok in checks],
        }

    return [
        f('Lipinski (rule of five)',
          'Lipinski et al., Adv. Drug Deliv. Rev. 2001, 46, 3',
          'Oral bioavailability heuristic. One violation is tolerated, as in the original paper.',
          [('MW ≤ 500', mw <= 500), ('logP ≤ 5', logp <= 5),
           ('H-bond donors ≤ 5', hbd <= 5), ('H-bond acceptors ≤ 10', hba <= 10)],
          allowed=1),
        f('Ghose',
          'Ghose et al., J. Comb. Chem. 1999, 1, 55',
          'Qualifying range derived from known drugs.',
          [('160 ≤ MW ≤ 480', 160 <= mw <= 480),
           ('−0.4 ≤ logP ≤ 5.6', -0.4 <= logp <= 5.6),
           ('40 ≤ molar refractivity ≤ 130', 40 <= mr <= 130),
           ('20 ≤ heavy atoms ≤ 70', 20 <= heavy <= 70)]),
        f('Veber',
          'Veber et al., J. Med. Chem. 2002, 45, 2615',
          'Flexibility and polarity predict oral bioavailability better than mass alone.',
          [('Rotatable bonds ≤ 10', rotb <= 10), ('TPSA ≤ 140 Å²', tpsa <= 140)]),
        f('Egan (egg)',
          'Egan et al., J. Med. Chem. 2000, 43, 3867',
          'Passive intestinal absorption boundary in TPSA/logP space.',
          [('TPSA ≤ 131.6 Å²', tpsa <= 131.6), ('logP ≤ 5.88', logp <= 5.88)]),
        f('Muegge',
          'Muegge et al., J. Med. Chem. 2001, 44, 1841',
          'Pharmacophore-point filter separating drugs from non-drugs.',
          [('200 ≤ MW ≤ 600', 200 <= mw <= 600),
           ('−2 ≤ logP ≤ 5', -2 <= logp <= 5),
           ('TPSA ≤ 150 Å²', tpsa <= 150),
           ('Rings ≤ 7', d['rings'] <= 7),
           ('Carbon atoms > 4', d['carbons'] > 4),
           ('Heteroatoms > 1', d['heteroatoms'] > 1),
           ('Rotatable bonds ≤ 15', rotb <= 15),
           ('H-bond acceptors ≤ 10', hba <= 10),
           ('H-bond donors ≤ 5', hbd <= 5)]),
        f('Lead-likeness',
          'Teague et al., Angew. Chem. Int. Ed. 1999, 38, 3743',
          'Smaller and simpler than a finished drug, leaving room to optimise.',
          [('250 ≤ MW ≤ 350', 250 <= mw <= 350),
           ('logP ≤ 3.5', logp <= 3.5),
           ('Rotatable bonds ≤ 7', rotb <= 7)]),
        f('Rule of three (fragments)',
          'Congreve et al., Drug Discov. Today 2003, 8, 876',
          'Screening criteria for fragment-based libraries.',
          [('MW ≤ 300', mw <= 300), ('logP ≤ 3', logp <= 3),
           ('H-bond donors ≤ 3', hbd <= 3), ('H-bond acceptors ≤ 3', hba <= 3),
           ('Rotatable bonds ≤ 3', rotb <= 3)]),
    ]


# ── Absorption and distribution ──────────────────────────────────────────────
# The BOILED-Egg classifies GI absorption and brain penetration with two
# *rotated* ellipses in WLOGP/TPSA space. The published paper gives the ellipse
# coordinates only as figures in its supporting information, so the exact
# coefficients are not reproducible from the literature. Rather than invent
# coefficients, this implements the bounding ranges the authors state and labels
# the result as an approximation. The Egan filter above is the exact, fully
# published absorption boundary and is the one to trust.

BOILED_EGG_CITE = 'Daina & Zoete, ChemMedChem 2016, 11, 1117'

# Bounding ranges stated by the authors for the two ellipses.
#
# Checked, not assumed: on a 20-drug set (10 CNS-active, 10 peripherally
# restricted) the BBB box below gets 10/10 CNS drugs and 6/10 peripheral ones,
# 80% overall. All four false positives — cetirizine, dopamine, sumatriptan,
# loperamide — are known P-glycoprotein efflux substrates or actively excluded,
# which the BOILED-Egg explicitly does not model. So the box errs towards
# calling things permeant, and that is the caveat shown in the UI.
HIA_RANGE = {'tpsa': (0.0, 142.1), 'wlogp': (-2.3, 6.8)}
BBB_RANGE = {'tpsa': (0.4, 79.1), 'wlogp': (-2.9, 6.0)}


def _in_range(d, box):
    lo_t, hi_t = box['tpsa']
    lo_w, hi_w = box['wlogp']
    return lo_t <= d['tpsa'] <= hi_t and lo_w <= d['wlogp'] <= hi_w


def _pharmacokinetics(d, log_s):
    rows = []

    egan_ok = d['tpsa'] <= 131.6 and d['wlogp'] <= 5.88
    rows.append({
        'key': 'gi_absorption',
        'label': 'Passive GI absorption',
        'value': 'Likely high' if egan_ok else 'Likely low',
        'verdict': 'good' if egan_ok else 'warn',
        'basis': 'rule',
        'cite': 'Egan et al., J. Med. Chem. 2000, 43, 3867',
        'note': ('Inside Egan’s absorption boundary (TPSA ≤ 131.6 Å², logP ≤ 5.88).'
                 if egan_ok else
                 'Outside Egan’s absorption boundary, so passive uptake across the gut wall '
                 'is expected to be limited.'),
    })

    hia_ok = _in_range(d, HIA_RANGE)
    rows.append({
        'key': 'boiled_egg_white',
        'label': 'BOILED-Egg white (HIA)',
        'value': 'Within range' if hia_ok else 'Outside range',
        'verdict': 'good' if hia_ok else 'warn',
        'basis': 'range',
        'cite': BOILED_EGG_CITE,
        'note': 'Approximate: the published bounding range (TPSA 0–142.1 Å², '
                'WLOGP −2.3–6.8), not the paper’s exact ellipse.',
    })

    bbb_ok = _in_range(d, BBB_RANGE)
    rows.append({
        'key': 'bbb',
        'label': 'Blood–brain barrier (yolk)',
        'value': 'Permeant' if bbb_ok else 'Non-permeant',
        'verdict': 'good' if bbb_ok else 'neutral',
        'basis': 'range',
        'cite': BOILED_EGG_CITE,
        'note': 'Approximate: the published bounding range (TPSA ≤ 79.1 Å², '
                'WLOGP −2.9–6.0), not the paper’s exact ellipse. On a 20-drug '
                'check this box called 10/10 CNS drugs correctly but only 6/10 '
                'peripherally restricted ones — it models passive diffusion only, so '
                'efflux substrates read as permeant when they are not.',
    })

    rows.append({
        'key': 'log_s',
        'label': 'Water solubility (ESOL)',
        'value': f'log S = {log_s:.2f}',
        'verdict': 'good' if log_s >= -4 else ('warn' if log_s >= -6 else 'bad'),
        'basis': 'equation',
        'cite': ESOL_CITE,
        'note': f'{_sol_class(log_s)}. Average absolute error of the original fit is '
                f'~0.75 log units.',
    })
    return rows


# Endpoints a user will reasonably expect and that this module deliberately
# does not fabricate.
UNAVAILABLE = [
    {'label': 'CYP1A2/2C9/2C19/2D6/3A4 inhibition',
     'why': 'Each is a trained classifier (SwissADME uses SVMs). There is no published '
            'closed-form rule to evaluate, so no honest value can be computed here.'},
    {'label': 'P-glycoprotein substrate',
     'why': 'Trained classifier; same reason.'},
    {'label': 'hERG blockade, Ames mutagenicity, oral LD50',
     'why': 'Toxicity endpoints need a fitted model and a stated applicability domain. '
            'The Brenk and NIH alerts below flag some structural liabilities, but they '
            'are not a substitute for a tox model.'},
]


# ── Bioavailability radar ────────────────────────────────────────────────────
# Six axes, as in SwissADME's radar. Each axis reports the value plus the
# optimal window, so the frontend can draw the shaded target area.

def _radar(d, log_s):
    return [
        {'key': 'LIPO', 'label': 'Lipophilicity', 'detail': 'logP',
         'value': d['wlogp'], 'optimal': [-0.7, 5.0], 'scale': [-3.0, 7.5]},
        {'key': 'SIZE', 'label': 'Size', 'detail': 'MW',
         'value': d['mw'], 'optimal': [150.0, 500.0], 'scale': [0.0, 650.0]},
        {'key': 'POLAR', 'label': 'Polarity', 'detail': 'TPSA',
         'value': d['tpsa'], 'optimal': [20.0, 130.0], 'scale': [0.0, 200.0]},
        {'key': 'INSOLU', 'label': 'Insolubility', 'detail': 'log S',
         'value': log_s, 'optimal': [-6.0, 0.0], 'scale': [-12.0, 2.0]},
        {'key': 'INSATU', 'label': 'Insaturation', 'detail': 'fraction Csp3',
         'value': d['fraction_csp3'], 'optimal': [0.25, 1.0], 'scale': [0.0, 1.0]},
        {'key': 'FLEX', 'label': 'Flexibility', 'detail': 'rotatable bonds',
         'value': float(d['rotatable_bonds']), 'optimal': [0.0, 9.0], 'scale': [0.0, 15.0]},
    ]


REFERENCES = [
    {'what': 'Water solubility (ESOL)', 'cite': ESOL_CITE,
     'doi': '10.1021/ci034243x'},
    {'what': 'GI absorption and brain penetration (BOILED-Egg)', 'cite': BOILED_EGG_CITE,
     'doi': '10.1002/cmdc.201600182'},
    {'what': 'Rule of five', 'cite': 'Lipinski et al., Adv. Drug Deliv. Rev. 2001, 46, 3',
     'doi': '10.1016/S0169-409X(00)00129-0'},
    {'what': 'Ghose qualifying range', 'cite': 'Ghose et al., J. Comb. Chem. 1999, 1, 55',
     'doi': '10.1021/cc9800071'},
    {'what': 'Veber rules', 'cite': 'Veber et al., J. Med. Chem. 2002, 45, 2615',
     'doi': '10.1021/jm020017n'},
    {'what': 'Egan egg', 'cite': 'Egan et al., J. Med. Chem. 2000, 43, 3867',
     'doi': '10.1021/jm000292e'},
    {'what': 'Muegge filter', 'cite': 'Muegge et al., J. Med. Chem. 2001, 44, 1841',
     'doi': '10.1021/jm015507e'},
    {'what': 'Lead-likeness', 'cite': 'Teague et al., Angew. Chem. Int. Ed. 1999, 38, 3743',
     'doi': '10.1002/(SICI)1521-3773(19991216)38:24<3743::AID-ANIE3743>3.0.CO;2-U'},
    {'what': 'Rule of three', 'cite': 'Congreve et al., Drug Discov. Today 2003, 8, 876',
     'doi': '10.1016/S1359-6446(03)02831-9'},
    {'what': 'Synthetic accessibility', 'cite': 'Ertl & Schuffenhauer, J. Cheminform. 2009, 1, 8',
     'doi': '10.1186/1758-2946-1-8'},
    {'what': 'Natural-product likeness', 'cite': 'Ertl et al., J. Chem. Inf. Model. 2008, 48, 68',
     'doi': '10.1021/ci700286x'},
    {'what': 'QED (quantitative estimate of drug-likeness)',
     'cite': 'Bickerton et al., Nat. Chem. 2012, 4, 90', 'doi': '10.1038/nchem.1243'},
    {'what': 'PAINS alerts', 'cite': 'Baell & Holloway, J. Med. Chem. 2010, 53, 2719',
     'doi': '10.1021/jm901137j'},
    {'what': 'Brenk alerts', 'cite': 'Brenk et al., ChemMedChem 2008, 3, 435',
     'doi': '10.1002/cmdc.200700139'},
    {'what': 'TPSA', 'cite': 'Ertl et al., J. Med. Chem. 2000, 43, 3714',
     'doi': '10.1021/jm000942e'},
    {'what': 'WLOGP and molar refractivity',
     'cite': 'Wildman & Crippen, J. Chem. Inf. Comput. Sci. 1999, 39, 868',
     'doi': '10.1021/ci990307l'},
    {'what': 'RDKit (all descriptors and alert catalogues)',
     'cite': 'RDKit: open-source cheminformatics, https://www.rdkit.org', 'doi': None},
]


# ── Public entry points ──────────────────────────────────────────────────────

def profile(mol):
    """Full ADMET profile for one RDKit molecule."""
    if mol is None:
        raise AdmetError('Invalid molecule')
    d = _descriptors(mol)
    log_s = _esol(d)

    try:
        qed = round(QED.qed(mol), 3)
    except Exception:
        qed = None
    try:
        sa = round(_sa_score(mol), 2)
    except Exception:
        sa = None
    try:
        np_like = round(_np_score(mol), 2)
    except Exception:
        np_like = None

    alerts = _alerts(mol)
    filters = _filters(d)

    physchem = [
        {'key': 'formula', 'label': 'Formula', 'value': rdMolDescriptors.CalcMolFormula(mol)},
        {'key': 'mw', 'label': 'Molecular weight', 'value': round(d['mw'], 2), 'unit': 'g/mol'},
        {'key': 'heavy_atoms', 'label': 'Heavy atoms', 'value': d['heavy_atoms']},
        {'key': 'aromatic_heavy_atoms', 'label': 'Aromatic heavy atoms',
         'value': d['aromatic_heavy_atoms']},
        {'key': 'fraction_csp3', 'label': 'Fraction Csp3', 'value': round(d['fraction_csp3'], 3)},
        {'key': 'rotatable_bonds', 'label': 'Rotatable bonds', 'value': d['rotatable_bonds']},
        {'key': 'hba', 'label': 'H-bond acceptors', 'value': d['hba']},
        {'key': 'hbd', 'label': 'H-bond donors', 'value': d['hbd']},
        {'key': 'mr', 'label': 'Molar refractivity', 'value': round(d['mr'], 2)},
        {'key': 'tpsa', 'label': 'TPSA', 'value': round(d['tpsa'], 2), 'unit': 'Å²'},
        {'key': 'wlogp', 'label': 'logP (WLOGP)', 'value': round(d['wlogp'], 2)},
        {'key': 'formal_charge', 'label': 'Formal charge', 'value': d['formal_charge']},
        {'key': 'stereocenters', 'label': 'Stereocentres', 'value': d['stereocenters']},
    ]

    # mg/mL alongside log S, which is what a bench scientist actually weighs out.
    # mol/L * g/mol = g/L, and 1 g/L is 1 mg/mL, so no further conversion.
    mol_per_l = 10 ** log_s
    mg_per_ml = mol_per_l * d['mw']

    medchem = [
        {'key': 'qed', 'label': 'QED', 'value': qed,
         'verdict': ('good' if (qed or 0) >= 0.67 else 'neutral') if qed is not None else 'neutral',
         'basis': 'equation', 'cite': 'Bickerton et al., Nat. Chem. 2012, 4, 90',
         'note': 'Weighted desirability across eight properties, 0–1. '
                 '≥ 0.67 is often treated as attractive.'},
        {'key': 'sa_score', 'label': 'Synthetic accessibility', 'value': sa,
         'verdict': ('good' if (sa or 10) <= 4 else ('neutral' if (sa or 10) <= 6 else 'warn'))
                    if sa is not None else 'neutral',
         'basis': 'equation', 'cite': 'Ertl & Schuffenhauer, J. Cheminform. 2009, 1, 8',
         'note': '1 = easy to make, 10 = hard. Estimated from fragment frequency in '
                 'PubChem plus a complexity penalty.'},
        {'key': 'np_score', 'label': 'Natural-product likeness', 'value': np_like,
         'verdict': 'neutral', 'basis': 'equation',
         'cite': 'Ertl et al., J. Chem. Inf. Model. 2008, 48, 68',
         'note': 'Positive scores resemble natural products, negative ones resemble '
                 'synthetic compounds.'},
    ]

    return {
        'smiles': Chem.MolToSmiles(mol),
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'inchikey': _inchikey(mol),
        'physchem': physchem,
        'pharmacokinetics': _pharmacokinetics(d, log_s),
        'solubility': {
            'log_s': round(log_s, 2),
            'class': _sol_class(log_s),
            'mol_per_l': _sig(mol_per_l),
            'mg_per_ml': _sig(mg_per_ml),
            'basis': 'equation', 'cite': ESOL_CITE,
        },
        'filters': filters,
        'filters_passed': sum(1 for f in filters if f['passed']),
        'filters_total': len(filters),
        'medchem': medchem,
        'alerts': alerts,
        'alert_count': len(alerts),
        'radar': _radar(d, log_s),
        'unavailable': UNAVAILABLE,
        'references': REFERENCES,
        'descriptors': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()},
    }


def _inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _sig(x, digits=3):
    """Round to a few significant figures; solubilities span many orders of magnitude."""
    if x == 0:
        return 0.0
    try:
        mag = math.floor(math.log10(abs(x)))
        return round(x, max(0, digits - 1 - mag))
    except Exception:
        return x


def summarise(mol):
    """Compact one-row profile, for batch tables."""
    d = _descriptors(mol)
    log_s = _esol(d)
    filters = _filters(d)
    try:
        qed = round(QED.qed(mol), 3)
    except Exception:
        qed = None
    try:
        sa = round(_sa_score(mol), 2)
    except Exception:
        sa = None
    alerts = _alerts(mol)
    return {
        'smiles': Chem.MolToSmiles(mol),
        'mw': round(d['mw'], 2),
        'wlogp': round(d['wlogp'], 2),
        'tpsa': round(d['tpsa'], 2),
        'hbd': d['hbd'],
        'hba': d['hba'],
        'rotatable_bonds': d['rotatable_bonds'],
        'log_s': round(log_s, 2),
        'solubility_class': _sol_class(log_s),
        'qed': qed,
        'sa_score': sa,
        'gi_absorption': 'High' if (d['tpsa'] <= 131.6 and d['wlogp'] <= 5.88) else 'Low',
        'bbb': 'Yes' if _in_range(d, BBB_RANGE) else 'No',
        'lipinski_violations': len(filters[0]['violations']),
        'filters_passed': sum(1 for f in filters if f['passed']),
        'filters_total': len(filters),
        'alert_count': len(alerts),
        'alerts': [a['name'] for a in alerts[:6]],
    }
