"""SmileBERTa API — accounts, saved workspace, ML models, molecule builder and docking."""
import json
import os
import secrets
import threading
import time
from functools import wraps

from flask import Flask, Response, jsonify, make_response, request, send_file
from flask_cors import CORS
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from werkzeug.security import check_password_hash, generate_password_hash

import db
import docking
import models
from chem_tools import (ChemError, analyze, builder_export, conformer_search, depict_svg, embed_3d,
                        mol_from_atoms, mol_from_smiles, mol_to_builder_json, optimize_geometry)
from combine_frag import combine_fragments
from fragpred import calculate_properties, cleanup_molecule_rdkit, load_fragment_model, predict_fragment_detailed
from predict_drug_classification import classify_detailed, drug_class_options, load_classification_model

try:
    import torch
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 2)))
except Exception:
    pass

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
CORS(app, origins='*', supports_credentials=False,
     allow_headers=['Content-Type', 'Authorization'], methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])

TOKEN_TTL_DAYS = 30
MAX_ITEM_BYTES = 6 * 1024 * 1024
ITEM_KINDS = {'molecule', 'fragment', 'classification', 'docking', 'structure', 'combination', 'note'}
STARTED_AT = time.time()

db.init_db()

# ── Model warm-up ────────────────────────────────────────────────────────────
# Load both transformer models in the background so the port binds immediately
# (Render health checks) while requests that need a model wait for it.

_models_state = {'classification': 'loading'}


def _warm_models():
    # Only the classifier is pre-loaded; the fragment model loads on first use (see models.py).
    try:
        load_classification_model()
        _models_state['classification'] = 'ready'
    except Exception as e:  # pragma: no cover
        _models_state['classification'] = 'error: ' + str(e)[:120]


if os.environ.get('SKIP_MODEL_WARMUP') != '1':
    threading.Thread(target=_warm_models, daemon=True).start()


# ── Helpers ──────────────────────────────────────────────────────────────────

def err(msg, code=400):
    return jsonify({'error': msg}), code


def body():
    return request.get_json(silent=True) or {}


def current_user():
    auth = request.headers.get('Authorization', '')
    if not auth.startswith('Bearer '):
        return None
    token = auth[7:].strip()
    if not token:
        return None
    with db.connect() as conn:
        row = conn.one('SELECT u.id, u.username, u.created_at, t.created_at AS token_created '
                       'FROM users u JOIN tokens t ON u.id = t.user_id WHERE t.token = ?', (token,))
    if not row:
        return None
    created = row.get('token_created')
    if created is not None:
        ts = created.timestamp() if hasattr(created, 'timestamp') else _parse_ts(created)
        if ts and time.time() - ts > TOKEN_TTL_DAYS * 86400:
            return None
    return row


def _parse_ts(s):
    try:
        return time.mktime(time.strptime(str(s)[:19], '%Y-%m-%d %H:%M:%S')) - time.timezone
    except Exception:
        return None


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user:
            return err('Not authenticated', 401)
        return fn(user, *args, **kwargs)
    return wrapper


def _iso(v):
    return v.isoformat() if hasattr(v, 'isoformat') else (str(v) if v is not None else None)


@app.errorhandler(413)
def too_large(_):
    return err('Upload too large (max 25 MB)', 413)


@app.route('/<path:_routes>', methods=['OPTIONS'])
def options_handler(_routes):
    return make_response('', 204)


# ── Status ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'uptime': round(time.time() - STARTED_AT),
        'models': _models_state,
        'models_in_memory': models.loaded(),
        'docking': docking.DOCKING_AVAILABLE,
        'database': 'postgres' if db.IS_POSTGRES else 'sqlite',
        'persistent_storage': db.IS_POSTGRES,
    })


# ── Auth ─────────────────────────────────────────────────────────────────────

def _issue_token(conn, user_id):
    token = secrets.token_urlsafe(32)
    conn.execute('INSERT INTO tokens (token, user_id) VALUES (?, ?)', (token, user_id))
    return token


@app.route('/register', methods=['POST'])
def register():
    data = body()
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    if len(username) < 3 or len(username) > 40:
        return err('Username must be 3–40 characters')
    if not all(c.isalnum() or c in '._-@' for c in username):
        return err('Username may only contain letters, numbers and . _ - @')
    if len(password) < 8:
        return err('Password must be at least 8 characters')
    try:
        with db.connect() as conn:
            uid = conn.insert('INSERT INTO users (username, password_hash) VALUES (?, ?)',
                              (username, generate_password_hash(password)))
            token = _issue_token(conn, uid)
    except Exception as e:
        if db.is_integrity_error(e):
            return err('That username is already taken', 409)
        raise
    return jsonify({'success': True, 'token': token, 'username': username})


@app.route('/login', methods=['POST'])
def login():
    data = body()
    username = str(data.get('username', '')).strip()
    password = str(data.get('password', ''))
    with db.connect() as conn:
        user = conn.one('SELECT id, password_hash FROM users WHERE username = ?', (username,))
        if not user or not check_password_hash(user['password_hash'], password):
            return err('Invalid username or password', 401)
        token = _issue_token(conn, user['id'])
    return jsonify({'token': token, 'username': username})


@app.route('/logout', methods=['POST'])
def logout():
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        with db.connect() as conn:
            conn.execute('DELETE FROM tokens WHERE token = ?', (auth[7:],))
    return jsonify({'success': True})


@app.route('/me', methods=['GET'])
@login_required
def me(user):
    with db.connect() as conn:
        counts = conn.all('SELECT kind, COUNT(*) AS n FROM items WHERE user_id = ? GROUP BY kind', (user['id'],))
        frags = conn.one('SELECT COUNT(*) AS n FROM fragment_library WHERE user_id = ?', (user['id'],))
    stats = {r['kind']: r['n'] for r in counts}
    stats['library'] = frags['n'] if frags else 0
    return jsonify({'username': user['username'], 'created_at': _iso(user.get('created_at')), 'stats': stats,
                    'persistent_storage': db.IS_POSTGRES})


@app.route('/account/password', methods=['POST'])
@login_required
def change_password(user):
    data = body()
    with db.connect() as conn:
        row = conn.one('SELECT password_hash FROM users WHERE id = ?', (user['id'],))
        if not check_password_hash(row['password_hash'], str(data.get('current', ''))):
            return err('Current password is incorrect', 403)
        new = str(data.get('new', ''))
        if len(new) < 8:
            return err('New password must be at least 8 characters')
        conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (generate_password_hash(new), user['id']))
    return jsonify({'success': True})


# ── Workspace items (saved work per account) ─────────────────────────────────

def _item_row(r, include_data=False):
    out = {'id': r['id'], 'kind': r['kind'], 'name': r['name'], 'summary': r.get('summary'),
           'created_at': _iso(r.get('created_at')), 'updated_at': _iso(r.get('updated_at'))}
    if include_data:
        try:
            out['data'] = json.loads(r['data']) if r.get('data') else None
        except ValueError:
            out['data'] = None
    return out


@app.route('/items', methods=['GET'])
@login_required
def list_items(user):
    kind = request.args.get('kind')
    with db.connect() as conn:
        if kind:
            rows = conn.all('SELECT id, kind, name, summary, created_at, updated_at FROM items '
                            'WHERE user_id = ? AND kind = ? ORDER BY updated_at DESC, id DESC', (user['id'], kind))
        else:
            rows = conn.all('SELECT id, kind, name, summary, created_at, updated_at FROM items '
                            'WHERE user_id = ? ORDER BY updated_at DESC, id DESC', (user['id'],))
    return jsonify({'items': [_item_row(r) for r in rows]})


def _validate_item(data, partial=False):
    kind = data.get('kind')
    if not partial and kind not in ITEM_KINDS:
        return None, 'Unknown item kind'
    name = str(data.get('name', '')).strip()[:120]
    if not partial and not name:
        return None, 'Name is required'
    payload = json.dumps(data.get('data')) if 'data' in data else None
    if payload and len(payload) > MAX_ITEM_BYTES:
        return None, 'Item too large to save (max 6 MB)'
    summary = data.get('summary')
    summary = json.dumps(summary)[:2000] if isinstance(summary, (dict, list)) else (str(summary)[:2000] if summary else None)
    return {'kind': kind, 'name': name, 'data': payload, 'summary': summary}, None


@app.route('/items', methods=['POST'])
@login_required
def create_item(user):
    item, e = _validate_item(body())
    if e:
        return err(e)
    with db.connect() as conn:
        n = conn.one('SELECT COUNT(*) AS n FROM items WHERE user_id = ?', (user['id'],))['n']
        if n >= 1000:
            return err('Workspace is full (1000 items). Delete some items first.', 409)
        iid = conn.insert('INSERT INTO items (user_id, kind, name, summary, data) VALUES (?, ?, ?, ?, ?)',
                          (user['id'], item['kind'], item['name'], item['summary'], item['data']))
    return jsonify({'success': True, 'id': iid})


@app.route('/items/<int:item_id>', methods=['GET'])
@login_required
def get_item(user, item_id):
    with db.connect() as conn:
        r = conn.one('SELECT * FROM items WHERE id = ? AND user_id = ?', (item_id, user['id']))
    if not r:
        return err('Not found', 404)
    return jsonify(_item_row(r, include_data=True))


@app.route('/items/<int:item_id>', methods=['PUT'])
@login_required
def update_item(user, item_id):
    item, e = _validate_item(body(), partial=True)
    if e:
        return err(e)
    sets, params = ['updated_at = CURRENT_TIMESTAMP'], []
    for col in ('name', 'summary', 'data'):
        if item[col]:
            sets.append(f'{col} = ?')
            params.append(item[col])
    with db.connect() as conn:
        cur = conn.execute(f'UPDATE items SET {", ".join(sets)} WHERE id = ? AND user_id = ?',
                           (*params, item_id, user['id']))
        if cur.rowcount == 0:
            return err('Not found', 404)
    return jsonify({'success': True})


@app.route('/items/<int:item_id>', methods=['DELETE'])
@login_required
def delete_item(user, item_id):
    with db.connect() as conn:
        conn.execute('DELETE FROM items WHERE id = ? AND user_id = ?', (item_id, user['id']))
    return jsonify({'success': True})


# ── Fragment library (kept for the fragmentation/combine pages) ──────────────

@app.route('/save_fragment', methods=['POST'])
@login_required
def save_fragment(user):
    data = body()
    smiles = data.get('smiles')
    if not smiles:
        return err('SMILES required')
    props = data.get('props')
    if isinstance(props, (dict, list)):
        props = json.dumps(props)
    with db.connect() as conn:
        if conn.one('SELECT id FROM fragment_library WHERE user_id = ? AND smiles = ?', (user['id'], smiles)):
            return err('Fragment already in library', 409)
        conn.execute('INSERT INTO fragment_library (user_id, smiles, img_data, props) VALUES (?, ?, ?, ?)',
                     (user['id'], smiles, data.get('img'), props))
    return jsonify({'success': True})


@app.route('/get_fragments', methods=['GET'])
@login_required
def get_fragments(user):
    with db.connect() as conn:
        rows = conn.all('SELECT smiles, img_data, props, created_at FROM fragment_library '
                        'WHERE user_id = ? ORDER BY created_at DESC, id DESC', (user['id'],))
    return jsonify({'fragments': [{'smiles': r['smiles'], 'img': r['img_data'], 'props': r['props'],
                                   'created_at': _iso(r['created_at'])} for r in rows]})


@app.route('/delete_fragment', methods=['DELETE'])
@login_required
def delete_fragment(user):
    with db.connect() as conn:
        conn.execute('DELETE FROM fragment_library WHERE user_id = ? AND smiles = ?', (user['id'], body().get('smiles')))
    return jsonify({'success': True})


# ── Molecule utilities ───────────────────────────────────────────────────────

def _smiles_mol():
    return mol_from_smiles(body().get('smiles'))


@app.route('/analyze', methods=['POST'])
def analyze_route():
    try:
        mol = _smiles_mol()
        return jsonify({'properties': analyze(mol), 'svg': depict_svg(mol)})
    except ChemError as e:
        return err(str(e))


@app.route('/depict', methods=['POST'])
def depict_route():
    try:
        data = body()
        mol = _smiles_mol()
        svg = depict_svg(mol, int(data.get('width', 420)), int(data.get('height', 320)), dark=bool(data.get('dark')))
        return Response(svg, mimetype='image/svg+xml')
    except ChemError as e:
        return err(str(e))


@app.route('/get_3d_structure', methods=['POST'])
def get_3d_structure_route():
    try:
        mol = embed_3d(_smiles_mol())
        return jsonify({'pdb': Chem.MolToPDBBlock(mol), 'sdf': Chem.MolToMolBlock(mol)})
    except ChemError as e:
        return err(str(e))


@app.route('/get_2d_structure', methods=['POST'])
def get_2d_structure_route():
    try:
        mol = _smiles_mol()
    except ChemError as e:
        return err(str(e))
    import io
    buf = io.BytesIO()
    Draw.MolToImage(mol, size=(400, 300)).save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/download_pdb', methods=['POST'])
def download_pdb():
    try:
        mol = embed_3d(_smiles_mol())
    except ChemError as e:
        return err(str(e))
    filename = os.path.basename(str(body().get('filename') or 'structure.pdb'))
    return Response(Chem.MolToPDBBlock(mol), mimetype='chemical/x-pdb',
                    headers={'Content-Disposition': f'attachment; filename="{filename}"'})


# ── Molecule builder (force fields) ──────────────────────────────────────────

@app.route('/builder/from_smiles', methods=['POST'])
def builder_from_smiles():
    try:
        mol = embed_3d(_smiles_mol())
        out = mol_to_builder_json(mol)
        out['properties'] = analyze(Chem.RemoveHs(mol))
        return jsonify(out)
    except ChemError as e:
        return err(str(e))


@app.route('/builder/optimize', methods=['POST'])
def builder_optimize():
    data = body()
    try:
        mol = mol_from_atoms(data.get('atoms'), data.get('bonds'))
        steps = max(10, min(int(data.get('steps', 2000)), 10000))
        result = optimize_geometry(mol, data.get('forcefield', 'MMFF94'), steps, int(data.get('frames', 30)))
        result.update(mol_to_builder_json(mol))
        try:
            result['properties'] = analyze(Chem.RemoveHs(mol))
        except Exception:
            result['properties'] = None
        return jsonify(result)
    except ChemError as e:
        return err(str(e))


@app.route('/builder/conformers', methods=['POST'])
def builder_conformers():
    data = body()
    try:
        if data.get('atoms'):
            mol = mol_from_atoms(data['atoms'], data.get('bonds'))
        else:
            mol = _smiles_mol()
        if Chem.RemoveHs(mol).GetNumHeavyAtoms() > 80:
            return err('Conformer search is limited to 80 heavy atoms')
        return jsonify(conformer_search(mol, data.get('n', 10), data.get('forcefield', 'MMFF94')))
    except ChemError as e:
        return err(str(e))


@app.route('/builder/export', methods=['POST'])
def builder_export_route():
    data = body()
    try:
        mol = mol_from_atoms(data.get('atoms'), data.get('bonds'))
        return jsonify({'text': builder_export(mol, data.get('format', 'sdf'))})
    except ChemError as e:
        return err(str(e))


# ── Fragment prediction & combination ────────────────────────────────────────

def _fragment_payload(smiles, protein):
    detail = predict_fragment_detailed(smiles, protein)
    cleaned = cleanup_molecule_rdkit(detail['fragment'])
    if not cleaned:
        return None, detail
    return cleaned, detail


@app.route('/predict_fragment', methods=['POST'])
def predict_fragment():
    data = body()
    smiles, protein = data.get('smiles'), data.get('protein')
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        return err('Please enter a valid SMILES string')
    cleaned, detail = _fragment_payload(smiles, protein)
    if not cleaned:
        return err('The model output could not be corrected into a valid fragment — try another molecule')
    try:
        frag_mol = embed_3d(mol_from_smiles(cleaned))
        pdb = Chem.MolToPDBBlock(frag_mol)
    except ChemError:
        pdb = None
    return jsonify({'fragment_smiles': cleaned, 'pdb': pdb, 'properties': calculate_properties(cleaned),
                    'analysis': analyze(mol_from_smiles(cleaned)), 'input_analysis': analyze(mol_from_smiles(smiles)),
                    'raw_output': detail['raw'], 'corrected': detail['corrected'],
                    'correction': detail['correction'], 'protein': protein})


@app.route('/batch_fragment', methods=['POST'])
def batch_fragment():
    data = body()
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return err('No SMILES provided')
    if len(smiles_list) > 50:
        return err('Maximum 50 SMILES per batch')
    results = []
    for s in smiles_list:
        s = str(s).strip()
        try:
            cleaned, detail = _fragment_payload(s, data.get('protein'))
            if cleaned:
                results.append({'input_smiles': s, 'fragment_smiles': cleaned, 'properties': calculate_properties(cleaned),
                                'corrected': detail['corrected'], 'error': None})
            else:
                results.append({'input_smiles': s, 'fragment_smiles': None, 'properties': None,
                                'error': 'Failed to generate fragment'})
        except Exception as e:
            results.append({'input_smiles': s, 'fragment_smiles': None, 'properties': None, 'error': str(e)})
    return jsonify({'results': results})


@app.route('/combine', methods=['POST'])
def combine():
    data = body()
    s1, s2 = str(data.get('smiles1') or ''), str(data.get('smiles2') or '')
    if Chem.MolFromSmiles(s1) is None or Chem.MolFromSmiles(s2) is None:
        return jsonify({'success': False, 'error': 'Both fragments must be valid SMILES'}), 400
    try:
        seen, out = set(), []
        for smi in combine_fragments(s1, s2):
            if smi in seen:
                continue
            seen.add(smi)
            mol = Chem.MolFromSmiles(smi)
            out.append({'smiles': smi, 'properties': calculate_properties(smi),
                        'analysis': analyze(mol) if mol else None})
        return jsonify({'success': True, 'combined_smiles': out})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


# ── Drug classification ──────────────────────────────────────────────────────

@app.route('/classify_smiles', methods=['POST'])
def classify_smiles():
    smiles = body().get('smiles')
    if not smiles:
        return err('No SMILES string provided')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return err('Please enter a valid SMILES string')
    try:
        result = classify_detailed(smiles)
        result['analysis'] = analyze(mol)
        return jsonify(result)
    except Exception as e:
        return err(str(e), 500)


@app.route('/batch_classify', methods=['POST'])
def batch_classify():
    smiles_list = body().get('smiles_list', [])
    if not smiles_list:
        return err('No SMILES provided')
    if len(smiles_list) > 100:
        return err('Maximum 100 SMILES per batch')
    results = []
    for s in smiles_list:
        s = str(s).strip()
        try:
            if Chem.MolFromSmiles(s) is None:
                raise ValueError('Invalid SMILES')
            r = classify_detailed(s)
            results.append({'smiles': s, 'prediction': r['prediction'], 'raw': r['raw'], 'error': None})
        except Exception as e:
            results.append({'smiles': s, 'prediction': None, 'error': str(e)})
    return jsonify({'results': results})


@app.route('/drug_classes', methods=['GET'])
def drug_classes():
    return jsonify({'classes': drug_class_options})


# ── Docking ──────────────────────────────────────────────────────────────────

@app.route('/dock', methods=['POST'])
def dock_submit():
    user = current_user()
    try:
        job_id = docking.submit(body(), user['id'] if user else None)
        return jsonify({'job_id': job_id})
    except docking.DockingError as e:
        return err(str(e))


@app.route('/dock/<job_id>', methods=['GET'])
def dock_status(job_id):
    job = docking.get_job(job_id)
    if not job:
        return err('Job not found (it may have expired or the server restarted)', 404)
    return jsonify(job)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5001)), threaded=True)
