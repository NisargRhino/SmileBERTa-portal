# training.py
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from tqdm import tqdm

##############################################################################
# CONFIG
##############################################################################
VIABLE_DRUGS_CSV = "/Users/nisargshah/Documents/cs/SmileBERTa-portal/ml-project/viable_drugs.csv"  # Input CSV with a "Viable_SMILES" column
OUTPUT_PKL = "viable_fp_data.pkl"                 # Stores fingerprints, SMILES, and cluster labels
CLUSTER_THRESHOLD = 0.5                           # Adjust the threshold for clustering
DENDROGRAM_PNG = "dendrogram.png"                 # File name for saving the dendrogram image

##############################################################################
# 1. Read CSV and compute fingerprints
##############################################################################
def read_viable_drugs(csv_path=VIABLE_DRUGS_CSV):
    """
    Reads the viable_drugs.csv which has a column 'Viable_SMILES'.
    Returns a list of SMILES strings.
    """
    df = pd.read_csv(csv_path, nrows=30000)
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
    # pdist-style condensed matrix: store only the upper triangle
    # We'll use tqdm to show a progress bar for the outer loop
    for i in tqdm(range(n), desc="Building Distance Matrix"):
        for j in range(i + 1, n):
            dist = tanimoto_distance(fps[i], fps[j])
            dists.append(dist)
    return dists

def hierarchical_clustering(fps, threshold=CLUSTER_THRESHOLD, method='average', 
                            plot_dendrogram=True):
    """
    Perform hierarchical clustering on the fingerprints with Tanimoto distance.
    If plot_dendrogram=True, displays and saves a dendrogram.
    Returns cluster labels and the linkage matrix.
    """
    dist_array = build_distance_matrix(fps)
    # Linkage on the distance array
    Z = linkage(dist_array, method=method)

    # Optionally plot the dendrogram
    if plot_dendrogram:
        plt.figure(figsize=(10, 6))
        dendrogram(
            Z,
            leaf_rotation=90.,     # Rotate x-axis labels
            leaf_font_size=8.0,    # Font size for x-axis labels
        )
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Compound Index")
        plt.ylabel("Distance")
        plt.tight_layout()
        plt.savefig(DENDROGRAM_PNG, dpi=300)
        plt.show()
        print(f"Dendrogram saved to {DENDROGRAM_PNG}.")

    # Use fcluster to get cluster labels by a distance threshold
    clusters = fcluster(Z, t=threshold, criterion='distance')
    return clusters, Z

##############################################################################
# 3. Main training routine
##############################################################################
def main():
    # 1) Read viable drug SMILES
    smiles_list = read_viable_drugs(VIABLE_DRUGS_CSV)

    # 2) Compute fingerprints with a progress bar
    fps = []
    valid_smiles = []
    print("Computing Morgan fingerprints...")
    for smi in tqdm(smiles_list, desc="Fingerprinting"):
        fp = compute_morgan_fp(smi)
        if fp:
            fps.append(fp)
            valid_smiles.append(smi)

    print(f"Loaded {len(valid_smiles)} viable SMILES with valid fingerprints.")

    # 3) Hierarchical Clustering (unsupervised)
    clusters, Z = hierarchical_clustering(
        fps, 
        threshold=CLUSTER_THRESHOLD, 
        method='average', 
        plot_dendrogram=True
    )
    print(f"Generated cluster labels for {len(clusters)} viable molecules.")

    # 4) Save data (fingerprints, SMILES, cluster labels)
    data_to_save = {
        'smiles': valid_smiles,
        'fingerprints': fps,
        'cluster_labels': clusters,
        'linkage_matrix': Z
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Saved unsupervised data (fingerprints, clusters) to {OUTPUT_PKL}.")

if __name__ == "__main__":
    main()
