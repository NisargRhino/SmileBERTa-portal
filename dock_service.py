"""Stand-alone docking API (AutoDock Vina).

Runs as its own Render service so Vina gets a full 512 MB instance instead of
sharing memory with the PyTorch models in app.py.
"""
import os
import time

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

import docking
import redock

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
# X-SmileBERTa-Client must be allowed or the browser's preflight blocks the
# header and every caller silently falls back to the shared IP bucket.
CORS(app, origins='*', allow_headers=['Content-Type', 'Authorization', 'X-SmileBERTa-Client'],
     methods=['GET', 'POST', 'OPTIONS'])
STARTED_AT = time.time()
# Validate the docking page's example structures once, in the background, so the
# first visitor to click "Validate this setup" gets a real result immediately.
redock.start_warm_thread()


@app.route('/<path:_routes>', methods=['OPTIONS'])
def options_handler(_routes):
    return make_response('', 204)


@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    # 'forwarded' reports whether the proxy is passing X-Forwarded-For. If it
    # were ever false in production, every caller would collapse onto the Docker
    # gateway address and share a single rate-limit bucket, locking everyone out
    # at the hourly cap. No address is disclosed, only whether the header exists.
    return jsonify({'status': 'ok', 'service': 'docking', 'docking': docking.DOCKING_AVAILABLE, 'vina': docking.VINA_VERSION,
                    'uptime': round(time.time() - STARTED_AT), 'rss_mb': docking.rss_mb(),
                    'forwarded': bool(request.headers.get('X-Forwarded-For')),
                    'redock_cache': redock.cache_stats(),
                    'queue': docking.queue_stats()})


@app.route('/dock', methods=['POST'])
def dock_submit():
    try:
        # No database here, so nobody can be authenticated: every caller is
        # rate-limited by client IP.
        return jsonify({'job_id': docking.submit(request.get_json(silent=True) or {},
                                                 ip=docking.client_ip(request),
                                                 cid=docking.client_id(request))})
    except docking.DockingError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/dock/validate', methods=['POST'])
def dock_validate():
    """Redock a structure's own co-crystallised ligand and measure the error."""
    data = request.get_json(silent=True) or {}
    try:
        job_id, cached = redock.start(data.get('pdb_id'), data.get('exhaustiveness', 4),
                                      data.get('num_poses', 9),
                                      ip=docking.client_ip(request),
                                      cid=docking.client_id(request))
        return jsonify({'job_id': job_id, 'cached': cached})
    except redock.RedockError as e:
        return jsonify({'error': str(e)}), 400
    except docking.DockingError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/dock/validate/<job_id>', methods=['GET'])
def dock_validate_status(job_id):
    st = redock.status(job_id)
    if not st:
        return jsonify({'error': 'Validation run not found (it may have expired or the '
                                 'server restarted)'}), 404
    return jsonify(st)


@app.route('/dock/<job_id>', methods=['GET'])
def dock_status(job_id):
    job = docking.get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found (it may have expired or the server restarted)'}), 404
    return jsonify(job)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5002)), threaded=True)
