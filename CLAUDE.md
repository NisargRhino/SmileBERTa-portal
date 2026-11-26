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

```bash
# Tests (redocking validation)
python3 tests/test_redock.py                      # unit tests, no network, no Vina
REDOCK_INTEGRATION=1 python3 tests/test_redock.py # + real PDB fetches and real docking runs
```

`frontend/assets/js/core.js` points at `http://localhost:5001` automatically when served from localhost, otherwise at `https://smileberta-api.onrender.com`. Override with `localStorage.sb_api`.

## Architecture

**Two Render services** (`render.yaml`): `smileberta-api` (Docker, Flask + PyTorch models, accounts) and `smileberta-web` (static site from `frontend/`). **Docking is a third service that does not run on Render** — `Dockerfile.dock` → `dock_service.py` (Vina only, no PyTorch) is deployed to a dedicated VPS; see `deploy/README.md` for how to ship to it, since pushing to GitHub does not. It is separate for two reasons: Vina pushed the ML API past the 512 MB free-tier limit (OOM restarts), and Vina is CPU-bound, where Render charges roughly ten times as much per core. The frontend calls `SB.DOCK_API` for `/dock` and `/dock/validate`.

**Backend modules:**
- `app.py` — all routes: auth (`/register`, `/login`, `/me`, `/account/password`), workspace items (`/items` CRUD, kinds: molecule, fragment, classification, combination, docking, structure), fragment library, `/analyze`, `/depict`, builder (`/builder/*`), ML (`/predict_fragment`, `/classify_smiles`, batch variants, `/combine`), docking (`/dock`, `/dock/<job_id>`, `/dock/validate`, `/dock/validate/<job_id>`), ADMET (`/admet`, `/admet/batch`, `/admet/references`), Hugging Face browser and notebook generator (`/hf/categories`, `/hf/models`, `/notebook`), and Bert (`/bert/status`, `/bert/chat`).
- `bert.py` — the agent. See the Bert section below.
- `admet.py` — RDKit-derived ADMET: absorption, ESOL solubility, BOILED-Egg, seven drug-likeness filters, PAINS/Brenk alerts. **Every number cites a published basis; endpoints that would need a trained model (CYP, hERG, Ames, LD50) are deliberately reported as unavailable rather than estimated.** Do not add invented predictions here.
- `notebooks.py` — generates Colab-ready notebooks wired to a user's CSV.
- `combine_frag.py` — amine/ester bridge linking with Lipinski and QED filters.
- `dock_service.py` — the standalone docking service deployed to the VPS (not imported by `app.py`). It owns `/dock`, `/dock/<job_id>` and the redocking-validation routes (`/dock/validate`, `/dock/validate/<job_id>`), which `app.py` mirrors so local development (where `SB.DOCK_API` is the local API) keeps working.
- `redock.py` — redocking validation ("Validate this setup" on `/docking`): pick the co-crystallised ligand, rebuild it from its chemical-component definition, dock it back into a box derived from the crystal ligand and measure symmetry-corrected heavy-atom RMSD against the crystal pose with `rdMolAlign.CalcRMS` (**never** `GetBestRMS` — superposing before measuring would answer the wrong question). Verdict bands are ≤2 Å reliable, ≤3 Å marginal, above that unreliable; **do not soften them, and do not add retry-until-it-passes logic** — an "unreliable" verdict is a successful run. The crystal coordinates are never used as the docking starting point: an independently embedded conformer is what makes the test real, and `start_rmsd` in the result records that it differed.
- `db.py` — Postgres when `DATABASE_URL` is set (production: Neon, set on `smileberta-api` in Render), SQLite otherwise (local dev only — Render's free disk is wiped on every sleep/redeploy). One fresh Postgres connection per request; never falls back to SQLite when `DATABASE_URL` is set (returns 503 until the DB is reachable). Usernames are stored lowercase.
- `models.py` — LRU cache for the two RoBERTa models. `MODEL_CACHE_SIZE=1` on the free tier (512 MB RAM) keeps one model resident. Do not int8-quantize: it changes ~15% of fragment outputs.
- `fragpred.py` / `predict_drug_classification.py` — inference. Invalid fragment SMILES → external corrector → nearest of `unique_smile5.csv`. Class text → nearest label in `drugclassoptions_final1.csv` by Levenshtein.
- `chem_tools.py` — RDKit analysis, depiction, MMFF94/UFF optimisation with animation frames, ETKDG conformer search.
- `docking.py` — Open Babel receptor/ligand PDBQT prep, AutoDock Vina subprocess in a background thread (single gunicorn worker; jobs are in memory), contact/H-bond analysis. The job queue is a priority queue: validation runs are submitted at priority 1 so they never delay someone's real docking, and cost half a slot against the hourly rate limit.

**Frontend:** vanilla HTML/JS, no build step. Shared design system `assets/css/styles.css`; shared runtime `assets/js/core.js` (window.SB: API client, auth, nav/footer injection, toasts/modals, saveItem, RDKit.js depiction, 3Dmol helpers). One script per page in `assets/js/`. Tools: fragmentation, drug_class, combine, builder, docking, studio, admet, models; plus workspace, login, pricing, credits, admin, terms and the learn pages.

## Bert (the agent)

`bert.py` is a Claude model wired to the platform's own Python functions — eight tools:
`analyze_molecule`, `predict_fragment`, `classify_drug`, `link_fragments`, `admet_profile`,
`dock`, `inspect_structure`, `search_papers` (Europe PMC). It plans multi-step work from one
plain-language request, runs the tools in order and reports back.

- **Tools call Python directly, never the platform's own HTTP API** — one process, no round
  trip, and no chance of the agent authenticating as another user.
- **Tool results are deliberately small.** The real endpoints return SVGs, conformers and full
  pose geometry; none of that helps a language model and all of it costs tokens. Keep new tools
  to numbers and identifiers, and let the UI link out for pictures.
- Every tool call is streamed to the browser as it happens, with arguments and result. `/bert/chat`
  is SSE; the browser parses it in `bert-chat.js`.
- **Only turns that reached the model are billed against the quota.** A rejected key or an
  unreachable API burns no tokens and must not consume somebody's daily allowance.

**Frontend:** `bert-chat.js` owns the conversation (sessionStorage-backed, shared between
surfaces, so the corner panel and the full page are one thread); `bert-launcher.js` is the
docked FAB injected on every page by `core.js`; `bert.js` + `bert.html` are the full page.
Bert is deliberately **not** in `SB.TOOLS` — he is the way to drive the other eight, not a ninth.

**Config** (environment, on `smileberta-api`): `ANTHROPIC_API_KEY` (required; users may also
supply their own via the `X-Anthropic-Key` header), `BERT_MODEL`, `BERT_EFFORT`,
`BERT_FREE_TURNS`, `BERT_MAX_TOOL_CALLS`, `BERT_MAX_DOCKS`, `DOCK_API`, `BERT_DOCK_TIMEOUT`.

**Cost:** the fixed prefix is ~2,100 tokens resent on every call in the loop and there is no
prompt caching yet, so a two-tool turn costs roughly 6¢ on `claude-opus-5-5`. Model and free-turn
cap are env vars for exactly this reason — changing them needs no deploy.

## Deployment notes

- **Do not rename `requirements-local.txt` back to `requirements.txt`** — Render auto-detects Python from it and breaks the static site.
- The Dockerfile installs `requirements-render.txt`, downloads the Vina 1.2.7 binary and bakes both Hugging Face models into the image.
- Static site sends `Cache-Control: no-cache`; when testing locally, hard-reload after JS edits.
- **The docking host does not auto-deploy.** Pushing to GitHub ships the API and the web site only; the VPS needs an explicit redeploy, and the script can silently do nothing. See `deploy/README.md`.
- **`smileberta-api` is near its memory ceiling.** It idles at ~404 MB and reaches ~565 MB once Bert's first request lazily imports the Anthropic SDK, against a 512 MB tier. It has been stable there, but assume no headroom: do not add a resident model or another heavyweight import without measuring `/health`'s `rss_mb` afterwards.
- `BERT_FREE_TURNS` was declared in `render.yaml` but never reached the service, so `smileberta-api` may not be blueprint-managed. Set Bert's environment variables in the Render dashboard and verify with `/bert/status`, rather than assuming a `render.yaml` edit will apply.
