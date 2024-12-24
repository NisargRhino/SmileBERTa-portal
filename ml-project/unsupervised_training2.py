# training.py
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

##############################################################################
# CONFIG
##############################################################################
VIABLE_DRUGS_CSV = "/Users/nisargshah/Documents/cs/SmileBERTa-portal/ml-project/viable_drugs.csv"         # Input CSV with a "Viable_SMILES" column
OUTPUT_PKL = "viable_fp_data.pkl"             # Stores fingerprints, SMILES, and cluster labels
CLUSTER_THRESHOLD = 0.5                       # Adjust the threshold for clustering

##############################################################################
# 1. Read CSV and compute fingerprints
##############################################################################
def read_viable_drugs(csv_path=VIABLE_DRUGS_CSV):
    """
    Reads the viable_drugs.csv which has a column 'Viable_SMILES'.
    Returns a list of SMILES strings.
    """
    df = pd.read_csv(csv_path)
    if 'Viable_SMILES' not in df.columns:
        raise ValueError("CSV must have a column 'Viable_SMILES'.")
    smiles_list = df['Viable_SMILES'].dropna().tolist()
    return smiles_list

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
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return fp

##############################################################################
# 2. Build distance matrix (1 - Tanimoto) and cluster
##############################################################################
def tanimoto_distance(fp1, fp2):
    return 1.0 - DataStructs.TanimotoSimilarity(fp1, fp2)

def build_distance_matrix(fps):
    """
    Given a list of fingerprints, build a condensed distance matrix
    using (1 - Tanimoto). This can be used by SciPy's linkage function.
    """
    n = len(fps)
    dists = []
    # pdist-style condensed matrix: store only upper triangle
    for i in range(n):
        for j in range(i + 1, n):
            dist = tanimoto_distance(fps[i], fps[j])
            dists.append(dist)
    return dists

def hierarchical_clustering(fps, threshold=CLUSTER_THRESHOLD, method='average'):
    """
    Perform hierarchical clustering on the fingerprints with Tanimoto distance.
    Returns cluster labels.
    """
    dist_array = build_distance_matrix(fps)
    # Linkage on the distance array
    Z = linkage(dist_array, method=method)
    # Use fcluster to get cluster labels by a distance threshold
    clusters = fcluster(Z, t=threshold, criterion='distance')
    return clusters

##############################################################################
# 3. Main training routine
##############################################################################
def main():
    # 1) Read viable drug SMILES
    smiles_list = read_viable_drugs(VIABLE_DRUGS_CSV)

    # 2) Compute fingerprints
    fps = []
    valid_smiles = []
    for smi in smiles_list:
        fp = compute_morgan_fp(smi)
        if fp:
            fps.append(fp)
            valid_smiles.append(smi)

    print(f"Loaded {len(valid_smiles)} viable SMILES with valid fingerprints.")

    # 3) Hierarchical Clustering (unsupervised)
    clusters = hierarchical_clustering(fps, threshold=CLUSTER_THRESHOLD, method='average')
    print(f"Generated cluster labels for {len(clusters)} viable molecules.")

    # 4) Save data (fingerprints, SMILES, cluster labels)
    data_to_save = {
        'smiles': valid_smiles,
        'fingerprints': fps,
        'cluster_labels': clusters
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved unsupervised data (fingerprints, clusters) to {OUTPUT_PKL}.")

if __name__ == "__main__":
    main()
