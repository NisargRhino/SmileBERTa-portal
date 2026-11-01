"""Therapeutic-class prediction with the SmileBERTa classification model.

The fine-tuned RoBERTa model emits a short class label as text. Because the
decoder is free-form, its output is snapped onto the closest of the known class
labels (drugclassoptions_final1.csv) by Levenshtein edit distance.
"""
import pandas as pd
import torch
from Levenshtein import distance as levenshtein_distance
from models import get_model, model_lock

CLASSIFICATION_MODEL = 'NisargRhino/drug-classification'

drug_class_options = pd.read_csv('./drugclassoptions_final1.csv')['name'].dropna().tolist()

def load_classification_model():
    return get_model(CLASSIFICATION_MODEL)


def rank_options(raw, options=None, k=5):
    options = options or drug_class_options
    scored = []
    for opt in options:
        d = levenshtein_distance(raw, opt)
        sim = 1 - d / max(len(raw), len(opt), 1)
        scored.append({'name': opt, 'distance': d, 'similarity': round(max(sim, 0.0), 3)})
    # The prediction is the label with the smallest edit distance (original SmileBERTa rule);
    # the remaining candidates are ordered by normalised similarity for display.
    scored.sort(key=lambda s: (s['distance'], -s['similarity']))
    winner, rest = scored[0], scored[1:]
    rest.sort(key=lambda s: (-s['similarity'], s['distance']))
    return ([winner] + rest)[:k]


def find_most_similar_option(predicted, options):
    return rank_options(predicted, options, k=1)[0]['name']


def classify_detailed(smiles, model=None, tokenizer=None, max_length=128):
    with model_lock():
        if model is None or tokenizer is None:
            model, tokenizer = load_classification_model()
        inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        raw = tokenizer.decode(predicted_ids[0], skip_special_tokens=True).strip()
    candidates = rank_options(raw)
    return {'prediction': candidates[0]['name'], 'raw': raw, 'candidates': candidates,
            'classes': candidates[0]['name'].split()}


def drug_class_smiles(smiles, model=None, tokenizer=None, max_length=128):
    return classify_detailed(smiles, model, tokenizer, max_length)['prediction']
