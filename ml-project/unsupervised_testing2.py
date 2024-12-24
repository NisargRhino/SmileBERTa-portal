# testing.py
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

##############################################################################
# CONFIG
##############################################################################
PICKLE_PATH = "viable_fp_data.pkl"                      # The data from training.py
CANCER_INHIBITORS_CSV = "cancer-inhibitors-augmented.csv" # Original CSV with 'SMILES' column
TOP_N = 50                                               # Number of top similar drugs to retrieve

##############################################################################
# 1. Load the unsupervised data (viable fingerprints, smiles)
##############################################################################
def load_training_data(pkl_path=PICKLE_PATH):
    """
    Loads data from the pickle file:
      - smiles: list of viable SMILES
      - fingerprints: list of RDKit fingerprint objects
      - cluster_labels: array of cluster labels
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

def mol_from_smiles(smiles):
    return Chem.MolFromSmiles(smiles) if smiles else None

def compute_morgan_fp(smiles, radius=2, nBits=2048):
    """
    Computes Morgan fingerprint for a SMILES string.
    Returns None if SMILES is invalid.
    """
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)

##############################################################################
# 2. For each query drug, compute Tanimoto to all viable drugs & pick top 50
##############################################################################
def find_top_similar_drugs(query_smiles, viable_smiles, viable_fps, top_n=TOP_N):
    """
    Given a query SMILES, compute Tanimoto to each viable drug fingerprint,
    and return the top N matches (as a list of tuples: [(similarity, smi), ...]).
    """
    query_fp = compute_morgan_fp(query_smiles)
    if query_fp is None:
        return []

    # Calculate similarities
    results = []
    for smi, fp in zip(viable_smiles, viable_fps):
        tanimoto = DataStructs.TanimotoSimilarity(query_fp, fp)
        results.append((tanimoto, smi))

    # Sort by descending similarity & return top N
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_n]

##############################################################################
# 3. Main testing routine
##############################################################################
def main():
    # 1) Load the unsupervised training data
    data = load_training_data(PICKLE_PATH)
    viable_smiles = data['smiles']
    viable_fps = data['fingerprints']

    # 2) Read the original augmented CSV with 'SMILES' column
    df = pd.read_csv(CANCER_INHIBITORS_CSV)
    if 'SMILES' not in df.columns:
        raise ValueError(f"{CANCER_INHIBITORS_CSV} must have a 'SMILES' column.")

    # 3) For each drug in the augmented CSV, find top 50 similar
    top_matches_per_drug = []
    for idx, row in df.iterrows():
        query_smi = str(row['SMILES']).strip()
        if not query_smi:
            continue
        top_matches = find_top_similar_drugs(query_smi, viable_smiles, viable_fps, top_n=TOP_N)
        
        # Example: store the best match or entire top 50
        # We'll store (query_smiles, [list of top 50 (similarity, viable_smiles)])
        top_matches_per_drug.append({
            'Query_SMILES': query_smi,
            'Top_Matches': top_matches  # a list of (similarity, viable_smiles)
        })

    # 4) Format the results: each row -> query SMILES + top 50 results
    # We'll flatten them into a DataFrame or just pick the best for demonstration
    # Example: create a DataFrame of query + best match
    # or write all 50 matches in a nested structure (dictionary or JSON).
    
    results_records = []
    for record in top_matches_per_drug:
        query_smi = record['Query_SMILES']
        top_list = record['Top_Matches']
        # We'll record them as multiple rows: one per best match
        for rank_i, (similarity, viable_smi) in enumerate(top_list, start=1):
            results_records.append({
                'Query_SMILES': query_smi,
                'Rank': rank_i,
                'Similarity': similarity,
                'Viable_SMILES': viable_smi
            })

    results_df = pd.DataFrame(results_records)
    results_df.to_csv("top50_similar_results.csv", index=False)
    print("Top 50 similarity results saved to top50_similar_results.csv")

if __name__ == "__main__":
    main()
