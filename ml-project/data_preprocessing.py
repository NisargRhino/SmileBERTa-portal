# data_preprocessing.py
import pubchempy as pcp
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def retrieve_smiles_from_name(drug_name: str) -> str:
    """
    Uses PubChemPy to fetch the canonical SMILES for a given drug name.
    If no result is found, returns an empty string or None.
    """
    drug_name = drug_name.strip()
    if not drug_name:
        return ""

    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            return compounds[0].canonical_smiles
        else:
            return ""
    except Exception:
        return ""

def compute_rdkit_descriptors(smiles: str):
    """
    Given a SMILES string, compute basic RDKit descriptors.
    Return a dict of descriptor values (MolWt, LogP, etc.).
    """
    if not smiles:
        return {
            'MolWeight': 0.0,
            'LogP': 0.0,
            'NumHDonors': 0,
            'NumHAcceptors': 0
        }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Invalid SMILES
        return {
            'MolWeight': 0.0,
            'LogP': 0.0,
            'NumHDonors': 0,
            'NumHAcceptors': 0
        }

    molwt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)

    return {
        'MolWeight': molwt,
        'LogP': logp,
        'NumHDonors': h_donors,
        'NumHAcceptors': h_acceptors
    }

def augment_dataset_with_descriptors(
    input_csv="cancer-inhibitors.csv",
    output_csv="cancer-inhibitors-augmented.csv"
):
    """
    1) Load the original CSV
    2) For each row, fetch SMILES from drug name (the 'Product' column)
    3) Compute descriptors, store them in new columns
    4) Save the augmented CSV
    """
    df = pd.read_csv(input_csv)

    smiles_list = []
    molw_list = []
    logp_list = []
    hd_list = []
    ha_list = []

    for idx, row in df.iterrows():
        drug_name = str(row['Product'])
        # 1. Get SMILES
        smiles = retrieve_smiles_from_name(drug_name)
        # 2. Compute descriptors
        desc = compute_rdkit_descriptors(smiles)

        smiles_list.append(smiles)
        molw_list.append(desc['MolWeight'])
        logp_list.append(desc['LogP'])
        hd_list.append(desc['NumHDonors'])
        ha_list.append(desc['NumHAcceptors'])

    # Create new columns in the DataFrame
    df['SMILES'] = smiles_list
    df['MolWeight'] = molw_list
    df['LogP'] = logp_list
    df['NumHDonors'] = hd_list
    df['NumHAcceptors'] = ha_list

    # Save to a new CSV
    df.to_csv(output_csv, index=False)
    print(f"Augmented dataset saved to: {output_csv}")

if __name__ == "__main__":
    # Example usage
    augment_dataset_with_descriptors(
        input_csv="/Users/nisargshah/Documents/cs/SmileBERTa-portal/ml-project/cancer-inhibitors.csv",
        output_csv="cancer-inhibitors-augmented.csv"
    )
