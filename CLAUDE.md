# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running locally

```bash
# Backend (port 5001)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-render.txt
python app.py                      # docking needs a `vina` binary on PATH (or VINA_BIN)

# Frontend (no build step)
python3 serve_frontend.py            # http://localhost:8080 (serves /builder as builder.html, like Render)
```

`frontend/assets/js/core.js` points at `http://localhost:5001` automatically when served from localhost, otherwise at `https://smileberta-api.onrender.com`. Override with `localStorage.sb_api`.

## Architecture

**Three Render services** (`render.yaml`): `smileberta-api` (Docker, Flask + PyTorch models, accounts), `smileberta-dock` (`Dockerfile.dock` → `dock_service.py`, Vina only, no PyTorch) and `smileberta-web` (static site from `frontend/`). Docking lives in its own service because the ML API sits near 400 MB and Vina pushed it past the 512 MB free-tier limit (OOM restarts). The frontend calls `SB.DOCK_API` for `/dock`.

**Backend modules:**
- `app.py` — all routes: auth (`/register`, `/login`, `/me`, `/account/password`), workspace items (`/items` CRUD, kinds: molecule, fragment, classification, combination, docking, structure), fragment library, `/analyze`, `/depict`, builder (`/builder/*`), ML (`/predict_fragment`, `/classify_smiles`, batch variants, `/combine`), docking (`/dock`, `/dock/<job_id>`).
- `db.py` — SQLite by default; Postgres when `DATABASE_URL` is set (falls back to SQLite if unreachable). SQLite resets on every Render redeploy.
- `models.py` — LRU cache for the two RoBERTa models. `MODEL_CACHE_SIZE=1` on the free tier (512 MB RAM) keeps one model resident. Do not int8-quantize: it changes ~15% of fragment outputs.
- `fragpred.py` / `predict_drug_classification.py` — inference. Invalid fragment SMILES → external corrector → nearest of `unique_smile5.csv`. Class text → nearest label in `drugclassoptions_final1.csv` by Levenshtein.
- `chem_tools.py` — RDKit analysis, depiction, MMFF94/UFF optimisation with animation frames, ETKDG conformer search.
- `docking.py` — Open Babel receptor/ligand PDBQT prep, AutoDock Vina subprocess in a background thread (single gunicorn worker; jobs are in memory), contact/H-bond analysis.

**Frontend:** vanilla HTML/JS. Shared design system `assets/css/styles.css`; shared runtime `assets/js/core.js` (window.SB: API client, auth, nav/footer injection, toasts/modals, saveItem, RDKit.js depiction, 3Dmol helpers). One script per page in `assets/js/`. Tools: fragmentation, drug_class, combine, builder, docking, studio; plus workspace, login, learn pages.

## Deployment notes

- **Do not rename `requirements-local.txt` back to `requirements.txt`** — Render auto-detects Python from it and breaks the static site.
- The Dockerfile installs `requirements-render.txt`, downloads the Vina 1.2.7 binary and bakes both Hugging Face models into the image.
- Static site sends `Cache-Control: no-cache`; when testing locally, hard-reload after JS edits.
