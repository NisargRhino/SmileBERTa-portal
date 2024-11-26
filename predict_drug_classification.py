from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import pandas as pd
from Levenshtein import distance as levenshtein_distance


unique_smiles_df = pd.read_csv('./drugclassoptions_final1.csv')
drug_class_options = unique_smiles_df['name'].tolist()
print(drug_class_options)

def find_most_similar_option(predicted_smiles_initial, options):
    min_distance = float('inf')
    predicted_smiles = None
    print("made it here first")
    for option in options:
        dist = levenshtein_distance(predicted_smiles_initial, option)
        if dist < min_distance:
            min_distance = dist
            predicted_smiles = option
    return predicted_smiles


def drug_class_smiles(smiles, model, tokenizer, max_length=128):
    inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    predicted_smiles_initial = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    predicted_smiles_final = find_most_similar_option(predicted_smiles_initial, drug_class_options)
    print("made it here")
    return predicted_smiles_final


# model_dc = RobertaForMaskedLM.from_pretrained('NisargRhino/drug-classification')
# tokenizer_dc = RobertaTokenizer.from_pretrained('NisargRhino/drug-classification')

# smiles = "CC(=O)C(C)c1ccc(CC(C)C)cc1"
# smiles_final = drug_class_smiles(smiles, model_dc, tokenizer_dc)
# print( smiles_final)

