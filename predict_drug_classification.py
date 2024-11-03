from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
import pandas as pd

# Load model and tokenizer
model = RobertaForMaskedLM.from_pretrained('NisargRhino/drug-classification')
tokenizer = RobertaTokenizer.from_pretrained('NisargRhino/drug-classification')
model.eval()

# Load unique tags for similarity comparison if needed
unique_tags_df = pd.read_csv('./drug_classification_data_df.csv')
unique_tags_list = unique_tags_df['tags'].tolist()

app = Flask(__name__)
CORS(app)

@app.route('/classify_smiles', methods=['POST'])
def classify_smiles():
    data = request.get_json()
    smiles = data.get('smiles')
    if not smiles:
        return jsonify({'error': 'No SMILES string provided'}), 400

    try:
        prediction = predict_fragment_smiles(smiles, model, tokenizer)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def predict_fragment_smiles(smiles, model, tokenizer, max_length=128):
    inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    predicted_smiles = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return predicted_smiles

if __name__ == '__main__':
    app.run(debug=True, port=5000)
