# data_preprocessing.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

def load_and_clean_data(input_csv: str, output_csv: str) -> pd.DataFrame:
    """
    Loads the CSV, cleans/standardizes SMILES, optionally combines meta fields,
    and saves a cleaned dataframe.
    """
    df = pd.read_csv(input_csv)
    
    # Basic cleaning: drop rows with missing SMILES
    df = df.dropna(subset=["SMILES"])
    df.reset_index(drop=True, inplace=True)
    
    cleaned_smiles = []
    for smi in df["SMILES"]:
        try:
            mol = Chem.MolFromSmiles(smi)
            can_smi = Chem.MolToSmiles(mol) if mol else None
            cleaned_smiles.append(can_smi)
        except:
            cleaned_smiles.append(None)
    
    df["SMILES_cleaned"] = cleaned_smiles
    df = df.dropna(subset=["SMILES_cleaned"])
    df.reset_index(drop=True, inplace=True)
    
    # Combine indications / targets into a single text field (MetaText) if columns exist
    meta_texts = []
    for idx, row in df.iterrows():
        indications = str(row["Indications"]) if "Indications" in df.columns else ""
        targets = str(row["Targets"]) if "Targets" in df.columns else ""
        combined = f"<INDICATION={indications}> <TARGET={targets}>"
        meta_texts.append(combined.strip())
    
    df["MetaText"] = meta_texts
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Cleaned data saved to {output_csv}")
    return df

if __name__ == "__main__":
    input_csv = "cancer-inhibitors-augmented.csv"
    output_csv = "cancer_inhibitors_cleaned.csv"
    load_and_clean_data(input_csv, output_csv)
