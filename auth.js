const AUTH_API = 'https://smileberta-api.onrender.com';

function getToken()    { return localStorage.getItem('auth_token'); }
function getUsername() { return localStorage.getItem('auth_username'); }
function isLoggedIn()  { return !!getToken(); }

function signOut() {
    const token = getToken();
    if (token) {
        fetch(AUTH_API + '/logout', {
            method: 'POST',
            headers: { 'Authorization': 'Bearer ' + token }
        }).catch(() => {});
    }
    localStorage.removeItem('auth_token');
    localStorage.removeItem('auth_username');
    window.location.href = 'login.html';
}

function authHeaders(extra) {
    const h = Object.assign({ 'Content-Type': 'application/json' }, extra || {});
    const token = getToken();
    if (token) h['Authorization'] = 'Bearer ' + token;
    return h;
}

function renderNavAuth(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (isLoggedIn()) {
        el.innerHTML =
            '<span class="nav-username">' + getUsername() + '</span>' +
            '<button class="nav-signout-btn" onclick="signOut()">Sign Out</button>';
    } else {
        el.innerHTML = '<a class="nav-signin-link" href="login.html">Sign In</a>';
    }
}

// ── Fragment library (backend when logged in, localStorage otherwise) ────────

async function saveFragment(smiles, img, props) {
    if (isLoggedIn()) {
        try {
            const res = await fetch(AUTH_API + '/save_fragment', {
                method: 'POST',
                headers: authHeaders(),
                body: JSON.stringify({ smiles, img, props })
            });
            const data = await res.json();
            if (!res.ok) return { ok: false, error: data.error };
            return { ok: true };
        } catch (e) {
            return { ok: false, error: e.message };
        }
    }
    let lib = JSON.parse(localStorage.getItem('fragmentLibrary') || '[]');
    if (lib.find(f => f.smiles === smiles)) return { ok: false, error: 'Fragment already in library' };
    lib.push({ smiles, img, props });
    localStorage.setItem('fragmentLibrary', JSON.stringify(lib));
    return { ok: true };
}

async function loadFragmentLibrary() {
    if (isLoggedIn()) {
        try {
            const res = await fetch(AUTH_API + '/get_fragments', { headers: authHeaders() });
            const data = await res.json();
            return data.fragments || [];
        } catch { return []; }
    }
    return JSON.parse(localStorage.getItem('fragmentLibrary') || '[]');
}

async function deleteFragment(smiles) {
    if (isLoggedIn()) {
        await fetch(AUTH_API + '/delete_fragment', {
            method: 'DELETE',
            headers: authHeaders(),
            body: JSON.stringify({ smiles })
        });
    } else {
        let lib = JSON.parse(localStorage.getItem('fragmentLibrary') || '[]');
        lib = lib.filter(f => f.smiles !== smiles);
        localStorage.setItem('fragmentLibrary', JSON.stringify(lib));
    }
}

// CSV parsing helper — handles quoted fields and multiple column names
function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/);
    if (!lines.length) return [];
    const header = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/"/g, ''));
    const smilesIdx = header.findIndex(h =>
        h === 'smiles' || h === 'smile' || h === 'canonical_smiles' || h === 'structure'
    );
    const dataStart = smilesIdx >= 0 ? 1 : 0;
    const results = [];
    for (let i = dataStart; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const cols = lines[i].split(',').map(c => c.trim().replace(/^"|"$/g, ''));
        const smiles = smilesIdx >= 0 ? cols[smilesIdx] : cols[0];
        if (smiles && smiles.length > 2) results.push(smiles);
    }
    return results;
}

function downloadCSV(rows, filename) {
    const csv = rows.map(r => r.map(c => '"' + String(c || '').replace(/"/g, '""') + '"').join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
}
