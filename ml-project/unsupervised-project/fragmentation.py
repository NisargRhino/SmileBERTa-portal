# fragmentation.py

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

def generate_fragments(input_csv: str, output_csv: str):
    """
    Reads cleaned data, fragments each SMILES using BRICS,
    and saves the fragment list to a new CSV.
    """
    df = pd.read_csv(input_csv)
    
    fragment_records = []
    for idx, row in df.iterrows():
        parent_id = row["Product"] if "Product" in df.columns else f"Mol_{idx}"
        smi = row["SMILES_cleaned"] if "SMILES_cleaned" in df.columns else None
        meta = row["MetaText"] if "MetaText" in df.columns else ""
        if not isinstance(smi, str) or not smi.strip():
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Break bonds via BRICS
        brokenMol = BRICS.BreakBRICSBonds(mol)
        # Now extract individual fragments as separate Mol objects
        frags = Chem.GetMolFrags(brokenMol, asMols=True)
        
        frag_smiles_list = [Chem.MolToSmiles(frag) for frag in frags]
        
        if not frag_smiles_list:
            # fallback if no fragments generated
            frag_smiles_list = [smi]
        
        for fsm in frag_smiles_list:
            fragment_records.append({
                "ParentID": parent_id,
                "ParentSMILES": smi,
                "FragmentSMILES": fsm,
                "MetaText": meta
            })
    
    frag_df = pd.DataFrame(fragment_records)
    frag_df.to_csv(output_csv, index=False)
    print(f"[INFO] Fragmentation data saved to {output_csv}")

if __name__ == "__main__":
    input_csv = "cancer_inhibitors_cleaned.csv"
    output_csv = "cancer_inhibitors_fragments.csv"
    generate_fragments(input_csv, output_csv)
