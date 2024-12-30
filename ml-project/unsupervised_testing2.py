# testing.py
import pandas as pd
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


from rdkit import Chem

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np
from minisom import MiniSom
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from argparse import ArgumentParser
import sys
import os
import logging
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem.Descriptors import qed
from pathlib import Path

sys.path.append("C:\\Users\\nisar\\cs\\ml3\\DeepDL\\src")
from models import RNNLM, GCNModel
import utils as UTILS
import matplotlib.pyplot as plt
import numpy as np


##############################################################################
# CONFIG
##############################################################################
PICKLE_PATH = "C:\\Users\\nisar\\cs\\ml3\\SmileBERTa-portal\\ml-project\\viable_fp_data.pkl"                      # The data from training.py
CANCER_INHIBITORS_CSV = "C:\\Users\\nisar\\cs\\ml3\\SmileBERTa-portal\\ml-project\\cancer-inhibitors-augmented.csv" # Original CSV with 'SMILES' column
TOP_N = 50                                               # Number of top similar drugs to retrieve

def read_viable_drugs(csv_path):
    """
    Reads the viable_drugs.csv which has a column 'Viable_SMILES'.
    Returns a list of SMILES strings.
    """
    df = pd.read_csv(csv_path, nrows=10000)
    if 'Viable_SMILES' not in df.columns:
        raise ValueError("CSV must have a column 'Viable_SMILES'.")
    smiles_list = df['Viable_SMILES'].dropna().tolist()
    return smiles_list

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
    query_smi_list = []
    for idx, row in df.iterrows():
        query_smi = str(row['SMILES']).strip()
        query_smi_list.append(query_smi)
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

    # Save list to a text file
    query_smi_list = [item for item in query_smi_list if item != 'nan']
    with open('query_smiles.smi', 'w') as file:
        for item in query_smi_list:
            file.write(f"{item}\n")



class QED_model (object):
        @staticmethod
        def test (smiles: str):
            mol = Chem.MolFromSmiles(smiles)
            return qed(mol)

def convert_csv_smiles_to_list(input_csv):
    df = pd.read_csv(input_csv)
    smiles_list = df['smiles'].tolist()
    return smiles_list

def main_rnnlm_gcn():
    
    # 1) Read viable drug SMILES
    ##smiles_list = read_viable_drugs('C:\\Users\\nisar\\cs\\ml3\\SmileBERTa-portal\\ml-project\\test_smiles.txt')
    smiles_list = convert_csv_smiles_to_list('C:\\Users\\nisar\\cs\\ml3\\SmileBERTa-portal\\drug_classification_data_df.csv')
    #smiles_list = ['CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)O)C)C)O)OC)C)C)C)OC']
    # 2) Compute fingerprints
    valid_smiles = []
    for smi in smiles_list:
        fp = compute_morgan_fp(smi)
        if fp:
            if len(smi) > 100:
                smi = smi[:100]
            valid_smiles.append(smi)

    print(f"Loaded {len(valid_smiles)} viable SMILES with valid fingerprints.")


    device = "cpu" #"cuda:0"
    
    if False :
        model = QED_model()
    else :
        config = OmegaConf.load('C:\\Users\\nisar\\cs\\ml3\\DeepDL\\test\\result\\rnn_worlddrug\\config.yaml')
        #config = OmegaConf.load('C:\\Users\\nisar\\cs\\ml3\\DeepDL\\test\\result\\gcn_worlddrug_zinc15\\config.yaml')
        
        model_architecture = config.model.model # RNNLM or GCNModel

        if model_architecture == 'RNNLM' :
            model = RNNLM.load_model('C:\\Users\\nisar\\cs\\ml3\\DeepDL\\test\\result\\rnn_worlddrug', device)
        elif model_architecture == 'GCNModel' :
            model = GCNModel.load_model('C:\\Users\\nisar\\cs\\ml3\\DeepDL\\test\\result\\gcn_worlddrug_zinc15', device)
        else :
            logging.warning("ERR: Not Allowed Model Architecture")
            exit(1)

    # Run
    scores = []
    for smiles in valid_smiles :
        if len(smiles) > 100:
            smiles = smiles[:100]
            # Validate SMILES
        if Chem.MolFromSmiles(smiles) is None:
            logging.warning(f'Invalid SMILES: {smiles}')
            continue
        print(smiles)
        score = model.test(smiles)
        scores.append(score)

        logging.info(f'{smiles},{score:.3f}')

    # Plot scores
    plt.plot(scores)
    plt.xlabel('SMILES Index')
    plt.ylabel('Score')
    plt.title('Scores of SMILES')
    plt.show()

    # Plot CDF of scores
    scores = np.array(scores)
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    plt.plot(sorted_scores, cdf)
    plt.xlabel('Score')
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function of Scores')
    plt.show()


if __name__ == "__main__":
    #main()
    main_rnnlm_gcn()



