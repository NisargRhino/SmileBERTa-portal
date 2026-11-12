"""Hugging Face model discovery + runnable notebook generation.

Two jobs:
  1. Proxy the public Hugging Face model API, narrowed to chemistry / drug-discovery
     models, so the browser gets one clean list instead of raw hub JSON.
  2. Emit a ready-to-run .ipynb wired to the user's own CSV columns, so "I found a
     model" becomes "I have working code" without writing boilerplate.

No Hugging Face token is needed: only public endpoints are used.
"""
import json
import urllib.parse
import urllib.request

HF_API = 'https://huggingface.co/api/models'
USER_AGENT = 'SmileBERTa-portal/1.0'
TIMEOUT = 15

# Curated starting points. The hub's own tags are too coarse to find chemistry
# models reliably, so each category carries its own search terms.
CATEGORIES = {
    'molecular-property': {
        'label': 'Molecular property prediction',
        'blurb': 'Predict activity, toxicity or physicochemical properties from SMILES.',
        'queries': ['ChemBERTa', 'molformer', 'molecular property', 'SMILES classification'],
    },
    'molecule-generation': {
        'label': 'Molecule generation',
        'blurb': 'Generate or complete SMILES strings for de novo design.',
        'queries': ['molgpt', 'SMILES generation', 'molecule generation', 'chemgpt'],
    },
    'protein': {
        'label': 'Protein language models',
        'blurb': 'Embed or annotate protein sequences (ESM, ProtBERT and friends).',
        'queries': ['esm2', 'protbert', 'protein language model', 'prot_t5'],
    },
    'reaction': {
        'label': 'Reactions & retrosynthesis',
        'blurb': 'Predict reaction products or retrosynthetic routes.',
        'queries': ['retrosynthesis', 'reaction prediction', 'rxn', 'molecular transformer'],
    },
    'embedding': {
        'label': 'Chemical embeddings',
        'blurb': 'Turn molecules into vectors for similarity search or downstream ML.',
        'queries': ['ChemBERTa', 'chemical embedding', 'mol2vec', 'smiles embedding'],
    },
}

# Models worth surfacing first: known-good, widely used, and actually loadable.
FEATURED = [
    'seyonec/ChemBERTa-zinc-base-v1',
    'DeepChem/ChemBERTa-77M-MLM',
    'DeepChem/ChemBERTa-10M-MTR',
    'ibm/MoLFormer-XL-both-10pct',
    'facebook/esm2_t12_35M_UR50D',
    'NisargRhino/drug-classification',
    'NisargRhino/protein-models',
]


class NotebookError(ValueError):
    pass


def _hf_get(params):
    url = f'{HF_API}?{urllib.parse.urlencode(params)}'
    req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT, 'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def _slim(m):
    """Keep only what the UI shows; hub payloads are large and noisy."""
    return {
        'id': m.get('modelId') or m.get('id'),
        'downloads': m.get('downloads') or 0,
        'likes': m.get('likes') or 0,
        'pipeline_tag': m.get('pipeline_tag'),
        'library': m.get('library_name'),
        'tags': [t for t in (m.get('tags') or []) if not t.startswith(('license:', 'region:', 'dataset:'))][:8],
        'updated': m.get('lastModified'),
    }


def search_models(query='', category=None, limit=24):
    """Search the hub. An explicit query wins; otherwise use the category's terms."""
    limit = max(1, min(int(limit), 60))
    queries = []
    if query and query.strip():
        queries = [query.strip()]
    elif category and category in CATEGORIES:
        queries = CATEGORIES[category]['queries']
    else:
        queries = ['ChemBERTa', 'molecular property', 'SMILES']

    seen, out = set(), []
    for q in queries:
        if len(out) >= limit:
            break
        try:
            batch = _hf_get({'search': q, 'limit': limit, 'sort': 'downloads',
                             'direction': -1, 'full': 'false'})
        except Exception:
            continue
        for m in batch:
            s = _slim(m)
            if s['id'] and s['id'] not in seen:
                seen.add(s['id'])
                out.append(s)
    out.sort(key=lambda m: m['downloads'], reverse=True)
    return out[:limit]


def model_detail(model_id):
    if not model_id or model_id.count('/') > 1:
        raise NotebookError('Invalid model id')
    try:
        m = _hf_get({'search': model_id, 'limit': 5, 'full': 'false'})
    except Exception as e:
        raise NotebookError(f'Could not reach Hugging Face: {e}')
    for cand in m:
        if (cand.get('modelId') or cand.get('id')) == model_id:
            return _slim(cand)
    raise NotebookError(f'Model "{model_id}" not found on the hub')


# ── Notebook generation ──────────────────────────────────────────────────────

def _md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.strip().split('\n')}


def _code(src):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': src.strip().split('\n')}


def _with_ids(cells):
    """nbformat 4.5 requires a unique id per cell, and warns loudly without one."""
    for i, c in enumerate(cells):
        c['id'] = f'cell-{i:02d}'
    return cells


TASKS = {
    'predict': 'Run predictions with a pretrained model',
    'embed': 'Embed molecules and train a baseline model',
    'finetune': 'Fine-tune the model on your own labels',
    'descriptors': 'Compute RDKit descriptors and drug-likeness',
}


def build_notebook(model_id, task='predict', smiles_col='smiles', label_col='label',
                   dataset_hint='your_data.csv', api_base=None):
    if task not in TASKS:
        raise NotebookError(f'Unknown task "{task}". Choose one of: {", ".join(TASKS)}')
    if not model_id and task != 'descriptors':
        raise NotebookError('A model id is required')
    model_id = (model_id or '').strip()
    smiles_col = (smiles_col or 'smiles').strip()
    label_col = (label_col or 'label').strip()
    csv = (dataset_hint or 'your_data.csv').strip()

    cells = [_md(f"""
# {TASKS[task]}

Generated by [SmileBERTa]({api_base or 'https://smileberta-web.onrender.com'}).
{f'Model: **`{model_id}`**' if model_id else ''}

This notebook is ready to run top to bottom. In Colab, upload `{csv}` with the file
browser on the left, or change `CSV_PATH` below to a URL.

**Expected CSV columns:** `{smiles_col}`{f' (molecules) and `{label_col}` (what you want to predict)' if task in ('finetune', 'embed') else ' (molecules)'}
"""), _code(f"""
# ── Setup ───────────────────────────────────────────────────────────────────
%pip install -q rdkit pandas scikit-learn {'transformers torch' if task != 'descriptors' else ''}

CSV_PATH   = "{csv}"
SMILES_COL = "{smiles_col}"
{f'LABEL_COL  = "{label_col}"' if task in ('finetune', 'embed') else ''}
{f'MODEL_ID   = "{model_id}"' if model_id else ''}
"""), _code(f"""
# ── Load and validate ───────────────────────────────────────────────────────
# Invalid SMILES are the most common cause of confusing errors later, so they
# are dropped up front and reported.
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

df = pd.read_csv(CSV_PATH)
print(f"Loaded {{len(df)}} rows, columns: {{list(df.columns)}}")
assert SMILES_COL in df.columns, f"No column named '{{SMILES_COL}}' - found {{list(df.columns)}}"

df["_mol"] = df[SMILES_COL].astype(str).apply(Chem.MolFromSmiles)
bad = df["_mol"].isna()
if bad.any():
    print(f"Dropping {{bad.sum()}} invalid SMILES, e.g. {{df.loc[bad, SMILES_COL].head(3).tolist()}}")
df = df[~bad].copy()
df["canonical_smiles"] = df["_mol"].apply(Chem.MolToSmiles)
print(f"{{len(df)}} valid molecules ready")
df.head()
""")]

    if task == 'descriptors':
        cells.append(_code("""
# ── RDKit descriptors + drug-likeness ───────────────────────────────────────
from rdkit.Chem import Descriptors, Crippen, QED, Lipinski

def profile(mol):
    return {
        "mol_weight": Descriptors.MolWt(mol),
        "clogp": Crippen.MolLogP(mol),
        "tpsa": Descriptors.TPSA(mol),
        "h_donors": Lipinski.NumHDonors(mol),
        "h_acceptors": Lipinski.NumHAcceptors(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "rings": Descriptors.RingCount(mol),
        "qed": QED.qed(mol),
    }

props = pd.DataFrame([profile(m) for m in df["_mol"]], index=df.index)
df = pd.concat([df.drop(columns=["_mol"]), props], axis=1)

# Lipinski's Rule of Five: how many of the four limits each molecule breaks.
df["lipinski_violations"] = (
    (df.mol_weight > 500).astype(int) + (df.clogp > 5).astype(int)
    + (df.h_donors > 5).astype(int) + (df.h_acceptors > 10).astype(int)
)
print(df["lipinski_violations"].value_counts().sort_index())
df.head()
"""))
        cells.append(_code("""
# ── Save ────────────────────────────────────────────────────────────────────
df.to_csv("descriptors_out.csv", index=False)
print("Wrote descriptors_out.csv")
"""))

    elif task == 'predict':
        cells.append(_code("""
# ── Load the model ──────────────────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Loaded {MODEL_ID} on {device}")
print(f"Output labels: {model.config.id2label}")
"""))
        cells.append(_code("""
# ── Predict in batches ──────────────────────────────────────────────────────
import numpy as np

BATCH = 32
preds, confs = [], []

for i in range(0, len(df), BATCH):
    chunk = df["canonical_smiles"].iloc[i:i + BATCH].tolist()
    enc = tokenizer(chunk, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
    preds.extend(probs.argmax(axis=1))
    confs.extend(probs.max(axis=1))
    print(f"  {min(i + BATCH, len(df))}/{len(df)}", end="\\r")

id2label = model.config.id2label
df["prediction"] = [id2label.get(int(p), str(p)) for p in preds]
df["confidence"] = np.round(confs, 4)
df.drop(columns=["_mol"]).to_csv("predictions.csv", index=False)
print(f"\\nWrote predictions.csv")
df[["canonical_smiles", "prediction", "confidence"]].head(10)
"""))

    elif task == 'embed':
        cells.append(_code("""
# ── Embed molecules ─────────────────────────────────────────────────────────
# Mean-pooled hidden states make a solid fixed-length fingerprint. Masked
# padding tokens are excluded so short molecules are not diluted.
import torch, numpy as np
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def embed(smiles_list, batch=32):
    out = []
    for i in range(0, len(smiles_list), batch):
        enc = tokenizer(smiles_list[i:i + batch], padding=True, truncation=True,
                        max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hidden * mask).sum(1) / mask.sum(1)).cpu().numpy())
        print(f"  {min(i + batch, len(smiles_list))}/{len(smiles_list)}", end="\\r")
    return np.vstack(out)

X = embed(df["canonical_smiles"].tolist())
print(f"\\nEmbeddings: {X.shape}")
"""))
        cells.append(_code("""
# ── Baseline model on the embeddings ────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

assert LABEL_COL in df.columns, f"No column named '{LABEL_COL}'"
y = df[LABEL_COL].values

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
print(classification_report(y_te, clf.predict(X_te)))
"""))

    elif task == 'finetune':
        cells.append(_code("""
# ── Prepare the dataset ─────────────────────────────────────────────────────
%pip install -q datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

assert LABEL_COL in df.columns, f"No column named '{LABEL_COL}'"
labels = sorted(df[LABEL_COL].astype(str).unique())
label2id = {l: i for i, l in enumerate(labels)}
print(f"{len(labels)} classes: {labels}")

df["_y"] = df[LABEL_COL].astype(str).map(label2id)
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["_y"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def to_ds(frame):
    ds = Dataset.from_dict({"text": frame["canonical_smiles"].tolist(),
                            "labels": frame["_y"].tolist()})
    return ds.map(lambda b: tokenizer(b["text"], truncation=True, max_length=128), batched=True)

train_ds, eval_ds = to_ds(train_df), to_ds(eval_df)
print(f"train {len(train_ds)} / eval {len(eval_ds)}")
"""))
        cells.append(_code("""
# ── Fine-tune ───────────────────────────────────────────────────────────────
# A CPU run is slow; in Colab pick Runtime > Change runtime type > GPU.
import numpy as np
from transformers import (AutoModelForSequenceClassification, TrainingArguments,
                          Trainer, DataCollatorWithPadding)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID, num_labels=len(labels),
    id2label={i: l for l, i in label2id.items()}, label2id=label2id)

def metrics(p):
    return {"accuracy": (np.argmax(p.predictions, axis=1) == p.label_ids).mean()}

trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir="out", num_train_epochs=3,
                           per_device_train_batch_size=16, per_device_eval_batch_size=32,
                           learning_rate=2e-5, eval_strategy="epoch",
                           save_strategy="no", logging_steps=25, report_to=[]),
    train_dataset=train_ds, eval_dataset=eval_ds,
    data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=metrics)

trainer.train()
print(trainer.evaluate())
"""))
        cells.append(_code("""
# ── Save ────────────────────────────────────────────────────────────────────
model.save_pretrained("my-model"); tokenizer.save_pretrained("my-model")
print("Saved to ./my-model")

# To publish it to the Hugging Face Hub:
#   from huggingface_hub import notebook_login; notebook_login()
#   model.push_to_hub("your-username/your-model-name")
#   tokenizer.push_to_hub("your-username/your-model-name")
"""))

    cells.append(_md(f"""
---

### Next steps on SmileBERTa

- **Dock your best candidates** against a protein target at `/docking`
- **Check drug-likeness** in bulk at `/drug_class`
- **Build and optimise in 3D** at `/builder`

Generated for model `{model_id or 'n/a'}` · task `{task}`.
"""))

    return {
        'cells': _with_ids(cells),
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python', 'version': '3.11'},
        },
        'nbformat': 4,
        'nbformat_minor': 5,
    }
