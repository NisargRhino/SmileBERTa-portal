from flask import Flask, request, jsonify, send_file, send_from_directory, make_response
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from fragpred import predict_fragment_smiles, cleanup_molecule_rdkit, calculate_properties, get_3d_structure
from combine_frag import combine_fragments
try:
    from docking import run_docking
    DOCKING_AVAILABLE = True
except ImportError:
    DOCKING_AVAILABLE = False
import os
import sqlite3
import secrets
import pandas as pd
from transformers import RobertaTokenizer, RobertaForMaskedLM
from predict_drug_classification import drug_class_smiles
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(app, origins="*", supports_credentials=False)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    return response

@app.route('/<path:routes>', methods=['OPTIONS'])
def options_handler(routes):
    return make_response('', 204)

# ── Database ────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'smileberta.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS tokens (
            token   TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS fragment_library (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            smiles     TEXT NOT NULL,
            img_data   TEXT,
            props      TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    conn.close()

init_db()

def get_user_from_token(req):
    auth = req.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None
    token = auth[7:]
    conn = get_db()
    row = conn.execute(
        'SELECT u.id, u.username FROM users u JOIN tokens t ON u.id = t.user_id WHERE t.token = ?',
        (token,)
    ).fetchone()
    conn.close()
    return row

# ── Auth ────────────────────────────────────────────────────────────────────

@app.route('/register', methods=['POST'])
def register():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400
    try:
        conn = get_db()
        conn.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                     (username, generate_password_hash(password)))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'Username already taken'}), 409

@app.route('/login', methods=['POST'])
def login():
    data = request.json or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    conn = get_db()
    user = conn.execute('SELECT id, password_hash FROM users WHERE username = ?', (username,)).fetchone()
    conn.close()
    if not user or not check_password_hash(user['password_hash'], password):
        return jsonify({'error': 'Invalid username or password'}), 401
    token = secrets.token_urlsafe(32)
    conn = get_db()
    conn.execute('INSERT INTO tokens (token, user_id) VALUES (?, ?)', (token, user['id']))
    conn.commit()
    conn.close()
    return jsonify({'token': token, 'username': username})

@app.route('/logout', methods=['POST'])
def logout():
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        token = auth[7:]
        conn = get_db()
        conn.execute('DELETE FROM tokens WHERE token = ?', (token,))
        conn.commit()
        conn.close()
    return jsonify({'success': True})

@app.route('/me', methods=['GET'])
def me():
    user = get_user_from_token(request)
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({'username': user['username']})

# ── Fragment Library ─────────────────────────────────────────────────────────

@app.route('/save_fragment', methods=['POST'])
def save_fragment():
    user = get_user_from_token(request)
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json or {}
    smiles   = data.get('smiles')
    img_data = data.get('img')
    props    = data.get('props')
    if not smiles:
        return jsonify({'error': 'SMILES required'}), 400
    conn = get_db()
    existing = conn.execute(
        'SELECT id FROM fragment_library WHERE user_id = ? AND smiles = ?', (user['id'], smiles)
    ).fetchone()
    if existing:
        conn.close()
        return jsonify({'error': 'Fragment already in library'}), 409
    conn.execute(
        'INSERT INTO fragment_library (user_id, smiles, img_data, props) VALUES (?, ?, ?, ?)',
        (user['id'], smiles, img_data, props)
    )
    conn.commit()
    conn.close()
    return jsonify({'success': True})

@app.route('/get_fragments', methods=['GET'])
def get_fragments():
    user = get_user_from_token(request)
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    conn = get_db()
    rows = conn.execute(
        'SELECT smiles, img_data, props FROM fragment_library WHERE user_id = ? ORDER BY created_at DESC',
        (user['id'],)
    ).fetchall()
    conn.close()
    return jsonify({'fragments': [{'smiles': r['smiles'], 'img': r['img_data'], 'props': r['props']} for r in rows]})

@app.route('/delete_fragment', methods=['DELETE'])
def delete_fragment():
    user = get_user_from_token(request)
    if not user:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.json or {}
    smiles = data.get('smiles')
    conn = get_db()
    conn.execute('DELETE FROM fragment_library WHERE user_id = ? AND smiles = ?', (user['id'], smiles))
    conn.commit()
    conn.close()
    return jsonify({'success': True})

# ── Molecular Tools ──────────────────────────────────────────────────────────

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
    smiles  = data.get('smiles')
    protein = data.get('protein')
    if not smiles:
        return jsonify({"error": "SMILES string is required"}), 400
    fragment_smiles = predict_fragment_smiles(smiles, protein)
    if not fragment_smiles:
        return jsonify({"error": "Invalid SMILES string could not be corrected, pick another SMILES"}), 400
    cleaned_fragment_smiles = cleanup_molecule_rdkit(fragment_smiles)
    if not cleaned_fragment_smiles:
        return jsonify({"error": "Failed to generate a valid fragment"}), 500
    mol = Chem.MolFromSmiles(cleaned_fragment_smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES string"}), 400
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol, nonBondedThresh=500.0)
    fragment_pdb = Chem.MolToPDBBlock(mol)
    properties = calculate_properties(cleaned_fragment_smiles)
    if not fragment_pdb:
        return jsonify({"error": "Failed to generate PDB for fragment"}), 500
    return jsonify({"fragment_smiles": cleaned_fragment_smiles, "pdb": fragment_pdb, "properties": properties})

@app.route('/download_pdb', methods=['POST'])
def download_pdb():
    data = request.json
    smiles   = data.get('smiles')
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
    with open(file_path, 'w') as f:
        f.write(pdb_block)
    return send_file(file_path, as_attachment=True, mimetype='chemical/x-pdb', download_name=filename)

@app.route('/combine', methods=['POST'])
def combine():
    frag1_smiles = str(request.json.get('smiles1'))
    frag2_smiles = str(request.json.get('smiles2'))
    try:
        combined_smiles_list = combine_fragments(frag1_smiles, frag2_smiles)
        combined_fragments_with_properties = []
        for smiles in combined_smiles_list:
            properties = calculate_properties(smiles)
            combined_fragments_with_properties.append({'smiles': smiles, 'properties': properties})
        return jsonify({'success': True, 'combined_smiles': combined_fragments_with_properties})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ── Drug Classification ──────────────────────────────────────────────────────

drug_class_options = pd.read_csv('./drugclassoptions_final1.csv')['name'].tolist()

model_dc = RobertaForMaskedLM.from_pretrained('NisargRhino/drug-classification')
tokenizer_dc = RobertaTokenizer.from_pretrained('NisargRhino/drug-classification')
model_dc.eval()

@app.route('/classify_smiles', methods=['POST'])
def classify_smiles():
    data = request.get_json()
    smiles = data.get('smiles')
    if not smiles:
        return jsonify({'error': 'No SMILES string provided'}), 400
    try:
        prediction = drug_class_smiles(smiles, model_dc, tokenizer_dc)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    data = request.json or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({'error': 'No SMILES provided'}), 400
    if len(smiles_list) > 100:
        return jsonify({'error': 'Maximum 100 SMILES per batch'}), 400
    results = []
    for smiles in smiles_list:
        s = smiles.strip()
        try:
            prediction = drug_class_smiles(s, model_dc, tokenizer_dc)
            results.append({'smiles': s, 'prediction': prediction, 'error': None})
        except Exception as e:
            results.append({'smiles': s, 'prediction': None, 'error': str(e)})
    return jsonify({'results': results})

@app.route('/batch_fragment', methods=['POST'])
def batch_fragment():
    data = request.json or {}
    smiles_list = data.get('smiles_list', [])
    protein     = data.get('protein')
    if not smiles_list:
        return jsonify({'error': 'No SMILES provided'}), 400
    if len(smiles_list) > 50:
        return jsonify({'error': 'Maximum 50 SMILES per batch'}), 400
    results = []
    for smiles in smiles_list:
        s = smiles.strip()
        try:
            fragment_smiles = predict_fragment_smiles(s, protein)
            cleaned = cleanup_molecule_rdkit(fragment_smiles)
            if cleaned:
                properties = calculate_properties(cleaned)
                results.append({'input_smiles': s, 'fragment_smiles': cleaned, 'properties': properties, 'error': None})
            else:
                results.append({'input_smiles': s, 'fragment_smiles': None, 'properties': None, 'error': 'Failed to generate fragment'})
        except Exception as e:
            results.append({'input_smiles': s, 'fragment_smiles': None, 'properties': None, 'error': str(e)})
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
