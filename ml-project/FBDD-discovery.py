import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS, Descriptors

# Helper Functions
def fragment_drug(smiles):
    """
    Fragment a drug molecule using BRICS.
    Returns a set of fragments (SMILES strings).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return set()  # Handle invalid SMILES
        fragments = BRICS.BRICSDecompose(mol)
        return set(fragments)
    except Exception as e:
        print(f"Error fragmenting SMILES {smiles}: {e}")
        return set()

def combine_fragments(fragment_smiles):
    """
    Combine BRICS fragments into new molecules.
    Returns a list of new molecule SMILES.
    """
    try:
        combined_molecules = set()
        fragment_mols = [Chem.MolFromSmiles(frag) for frag in fragment_smiles if Chem.MolFromSmiles(frag)]
        
        for mol1 in fragment_mols:
            for mol2 in fragment_mols:
                if mol1 != mol2:
                    try:
                        combined = Chem.CombineMols(mol1, mol2)
                        combined_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(combined)))
                        combined_molecules.add(combined_smiles)
                    except Exception as e:
                        print(f"Error combining fragments {mol1} and {mol2}: {e}")
        return list(combined_molecules)
    except Exception as e:
        print(f"Error combining fragments: {e}")
        return []

def passes_lipinski(smiles):
    """
    Check if a molecule passes Lipinski's Rule of Five.
    Returns True if all rules are satisfied, otherwise False.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        return (
            mw <= 500 and  # Molecular weight ≤ 500
            logp <= 5 and  # LogP ≤ 5
            h_donors <= 5 and  # ≤ 5 hydrogen bond donors
            h_acceptors <= 10  # ≤ 10 hydrogen bond acceptors
        )
    except Exception as e:
        print(f"Error evaluating Lipinski's rule for SMILES {smiles}: {e}")
        return False

def process_csv(input_csv, fragments_csv, final_csv):
    """
    Main function to:
    1. Fragment drugs using BRICS.
    2. Recombine fragments into new drugs.
    3. Filter new drugs using Lipinski's Rule.
    4. Save results to CSV files.
    """
    # Load the input CSV
    df = pd.read_csv(input_csv)

    # Check if 'SMILES' column exists
    if 'SMILES' not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")

    # Step 1: Fragment all drugs
    all_fragments = set()
    fragment_map = {}
    for idx, row in df.iterrows():
        smiles = str(row['SMILES']).strip()
        if smiles:
            fragments = fragment_drug(smiles)
            all_fragments.update(fragments)
            fragment_map[smiles] = list(fragments)

    print(f"Number of unique fragments: {len(all_fragments)}")  # Debug

    # Save fragments to a CSV
    fragments_df = pd.DataFrame([(smiles, frag) for smiles, frags in fragment_map.items() for frag in frags], 
                                columns=["Original_SMILES", "Fragment"])
    fragments_df.to_csv(fragments_csv, index=False)
    print(f"Fragments saved to {fragments_csv}")

    # Step 2: Combine fragments to make new drugs
    new_drugs = combine_fragments(all_fragments)
    print(f"Number of new drugs generated: {len(new_drugs)}")  # Debug

    # Step 3: Filter new drugs using Lipinski's Rule
    viable_drugs = []
    for drug in new_drugs:
        if passes_lipinski(drug):
            viable_drugs.append(drug)
        else:
            print(f"Failed Lipinski: {drug}")  # Debug failed drugs

    print(f"Number of viable drugs: {len(viable_drugs)}")  # Debug

    # Save viable drugs to a CSV
    viable_drugs_df = pd.DataFrame(viable_drugs, columns=["Viable_SMILES"])
    viable_drugs_df.to_csv(final_csv, index=False)
    print(f"Viable drugs saved to {final_csv}")

# Main Execution
if __name__ == "__main__":
    # Input/output paths
    input_csv = "/Users/nisargshah/Documents/cs/SmileBERTa-portal/ml-project/cancer-inhibitors-augmented.csv"  # Input CSV with SMILES column
    fragments_csv = "/Users/nisargshah/Documents/cs/SmileBERTa-portal/ml-project/drug_fragments.csv"  # Intermediate CSV with fragments
    final_csv = "/Users/nisargshah/Documents/cs/SmileBERTa-portal/ml-project/viable_drugs.csv"  # Output CSV with viable drugs

    # Process the CSV
    process_csv(input_csv, fragments_csv, final_csv)
