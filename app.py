from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from fragpred import predict_fragment_smiles, cleanup_molecule_rdkit, calculate_properties, get_3d_structure
from combine_frag import combine_fragments
from docking import run_docking
import os
import pandas as pd
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template
from flask_cors import CORS
from flask import make_response
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import io
import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch
from Levenshtein import distance as levenshtein_distance
from predict_drug_classification import drug_class_smiles

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type, Accept, Origin, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

@app.route('/*', methods=['OPTIONS'])
def options_handler(routes):
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'X-Requested-With, Content-Type, Accept, Origin, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

# @app.route('/score', methods=['POST'])
# def score_compound():
#     try:
#         data = request.get_json()
#         smiles = data.get('smiles')
        
#         if not smiles:
#             return jsonify({'error': 'No SMILES string provided'}), 400
        
#         score = run_docking(smiles)
#         return jsonify({'smiles': smiles, 'score': score})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/get_3d_structure', methods=['POST'])
def get_3d_structure_route():
    data = request.json
    smiles = data.get('smiles')

    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    pdb_block = Chem.MolToPDBBlock(mol)

    return jsonify({"pdb": pdb_block})

@app.route('/get_2d_structure', methods=['POST'])
def get_2d_structure_route():
    data = request.json
    smiles = data.get('smiles')
    #smiles = 'CC=C(C)C(=O)OC1C(C)=CC23C(=O)C(C=C(COC(C)=O)C(O)C12O)C1C(CC3C)C1(C)C'
    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400

    img = Draw.MolToImage(mol)
    img_path = os.path.join(os.getcwd(), 'molecule.png')
    img.save(img_path)

    return send_file(img_path, mimetype='image/png')

@app.route('/predict_fragment', methods=['POST'])
def predict_fragment():
    data = request.json
    smiles = data.get('smiles')
    protein = data.get('protein')
    print("smiles----:" + smiles)
    print("protein----:" + protein)

    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400

    fragment_smiles = predict_fragment_smiles(smiles, protein)
    print("reached here 1 !!!!")
    cleaned_fragment_smiles = cleanup_molecule_rdkit(fragment_smiles)
    print("reached here 2 !!!!")

    if not cleaned_fragment_smiles:
        return jsonify({"error": "Failed to generate a valid fragment"}), 500
    mol = Chem.MolFromSmiles(cleaned_fragment_smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    fragment_pdb = Chem.MolToPDBBlock(mol)
    #fragment_pdb = get_3d_structure(cleaned_fragment_smiles)
    properties = calculate_properties(cleaned_fragment_smiles)

    if not fragment_pdb:
        return jsonify({"error": "Failed to generate PDB for fragment"}), 500

    return jsonify({
        "fragment_smiles": cleaned_fragment_smiles,
        "pdb": fragment_pdb,
        "properties": properties
    })

@app.route('/download_pdb', methods=['POST'])
def download_pdb():
    data = request.json
    smiles = data.get('smiles')
    filename = data.get('filename', 'structure.pdb')

    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400

    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    pdb_block = Chem.MolToPDBBlock(mol)
    file_path = os.path.join(os.getcwd(), filename)

    with open(file_path, 'w') as file:
        file.write(pdb_block)

    return send_file(file_path, as_attachment=True, mimetype='chemical/x-pdb', download_name=filename)

@app.route('/combine', methods=['POST'])
def combine():
    frag1_smiles = str(request.json.get('smiles1'))
    frag2_smiles = str(request.json.get('smiles2'))
    print(f'Received SMILES 1: {frag1_smiles}')
    print(f'Received SMILES 2: {frag2_smiles}')
    try:
        combined_smiles_list = combine_fragments(frag1_smiles, frag2_smiles)
        print(combined_smiles_list)

        combined_fragments_with_properties = []
        for smiles in combined_smiles_list:
            properties = calculate_properties(smiles)
            combined_fragments_with_properties.append({
                'smiles': smiles,
                'properties': properties
            })
            print(combined_fragments_with_properties)

        return jsonify({'success': True, 'combined_smiles': combined_fragments_with_properties})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
"""
def predict_drug_classification(smiles, model, tokenizer, max_length=128):
    inputs = tokenizer(smiles, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    predicted_smiles = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return predicted_smiles
"""
drug_class_options = pd.read_csv('./drugclassoptions_final1.csv')['name'].tolist()

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


model_dc = RobertaForMaskedLM.from_pretrained('NisargRhino/drug-classification')
tokenizer_dc = RobertaTokenizer.from_pretrained('NisargRhino/drug-classification')
model_dc.eval()
@app.route('/classify_smiles', methods=['POST'])
def classify_smiles():
    print("reached here 1")
    data = request.get_json()
    smiles = data.get('smiles')
    print("reached here 2")
    if not smiles:
        return jsonify({'error': 'No SMILES string provided'}), 400
    
    print("reached here 3")
    try:
        prediction = drug_class_smiles(smiles, model_dc, tokenizer_dc)
        print("reached here 4")
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)  
