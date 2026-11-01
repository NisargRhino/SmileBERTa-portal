"""Stand-alone docking API (AutoDock Vina).

Runs as its own Render service so Vina gets a full 512 MB instance instead of
sharing memory with the PyTorch models in app.py.
"""
import os
import time

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

import docking

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
CORS(app, origins='*', allow_headers=['Content-Type', 'Authorization'], methods=['GET', 'POST', 'OPTIONS'])
STARTED_AT = time.time()


@app.route('/<path:_routes>', methods=['OPTIONS'])
def options_handler(_routes):
    return make_response('', 204)


@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'docking', 'docking': docking.DOCKING_AVAILABLE, 'vina': docking.VINA_VERSION,
                    'uptime': round(time.time() - STARTED_AT), 'rss_mb': docking.rss_mb()})


@app.route('/dock', methods=['POST'])
def dock_submit():
    try:
        return jsonify({'job_id': docking.submit(request.get_json(silent=True) or {})})
    except docking.DockingError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/dock/<job_id>', methods=['GET'])
def dock_status(job_id):
    job = docking.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found (it may have expired or the server restarted)'}), 404
    return jsonify(job)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)), threaded=True)
