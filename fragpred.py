"""Fragment prediction with the SmileBERTa fragment model (RoBERTa, ChemBERTa-initialised).

The model is used as a position-wise "translator": the drug SMILES is tokenised,
every output position is decoded with argmax, and the resulting token sequence is
read back as the fragment SMILES. Invalid outputs are repaired by a SMILES
corrector service, falling back to the closest valid reference fragment.
"""
import Levenshtein
import pandas as pd
import requests
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from requests.exceptions import RequestException
from models import get_model, model_lock

FRAGMENT_MODEL = 'NisargRhino/protein-models'
CORRECTOR_URL = 'https://smiles-corrector-1.onrender.com/correct'

unique_smiles_list = pd.read_csv('unique_smile5.csv')['SMILES'].dropna().tolist()

def load_fragment_model():
    """Load (or fetch from the shared cache) the fragment model."""
    return get_model(FRAGMENT_MODEL)


def cleanup_molecule_rdkit(smiles):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol)


def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        'molecular_weight': round(Descriptors.MolWt(mol), 2),
        'log_p': round(Descriptors.MolLogP(mol), 2),
        'hydrogen_bond_donors': Descriptors.NumHDonors(mol),
        'hydrogen_bond_acceptors': Descriptors.NumHAcceptors(mol),
        'tpsa': round(Descriptors.TPSA(mol), 2),
    }


def tanimoto_similarity(smiles1, smiles2):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        return 0.0
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def string_similarity(smiles1, smiles2):
    max_len = max(len(smiles1), len(smiles2))
    if max_len == 0:
        return 1.0
    return 1 - Levenshtein.distance(smiles1, smiles2) / max_len


def is_valid_smiles(smiles):
    return bool(smiles) and Chem.MolFromSmiles(smiles) is not None


def closest_reference_smiles(predicted_smiles):
    """Offline fallback: nearest valid fragment from the reference set by edit distance."""
    best, best_sim = None, -1.0
    for s in unique_smiles_list:
        sim = string_similarity(predicted_smiles, s)
        if sim > best_sim:
            best, best_sim = s, sim
    return best


def correct_smiles_remote(predicted_smiles):
    try:
        r = requests.post(CORRECTOR_URL, json={'smiles': predicted_smiles}, timeout=45)
        if r.status_code == 200:
            corrected = r.json().get('corrected')
            if is_valid_smiles(corrected):
                return corrected
    except (RequestException, ValueError):
        pass
    return None


def predict_fragment_detailed(smiles, protein=None, max_length=128):
    """Return the fragment plus provenance (raw model output, whether it was corrected)."""
    with model_lock():
        model, tokenizer = load_fragment_model()
        inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        raw = tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
    if is_valid_smiles(raw):
        return {'fragment': raw, 'raw': raw, 'corrected': False, 'correction': None}
    fixed = correct_smiles_remote(raw)
    if fixed:
        return {'fragment': fixed, 'raw': raw, 'corrected': True, 'correction': 'corrector-model'}
    fixed = closest_reference_smiles(raw)
    return {'fragment': fixed, 'raw': raw, 'corrected': True, 'correction': 'nearest-reference'}


def predict_fragment_smiles(smiles, protein=None, max_length=128):
    return predict_fragment_detailed(smiles, protein, max_length)['fragment']
