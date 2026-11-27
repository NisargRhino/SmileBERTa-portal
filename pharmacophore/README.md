# Pharmacophore corpus

Protein–ligand complexes annotated with the interactions each ligand atom
actually makes, built to replace docking scores as the fragment model's
supervision. See `PHARMACOPHORE_DATA_TASK.md` for the spec and
`BENCHMARK.md` for why the old approach is being replaced.

## State as of 2026-11-27, 00:25

**Full pipeline built and running end to end.** Corpus build is in progress under
a watchdog.

| | |
| --- | --- |
| Candidate entries | 139,597 (X-ray, ≤ 2.5 Å, has a non-polymer entity) |
| Build rate | ~105 entries/min, 6 concurrent batches |
| Projected full run | ~22 h |
| Records | `corpus/v1/records/*.jsonl` (gitignored) |
| Health log | `corpus/v1/health.log` — a snapshot every 5 minutes |

Validated against known binding modes. **3PTB** gives `Cationic ASP189 2.92 Å`,
the benzamidine salt bridge. **4EY7** gives face-to-face π-stacking with
`TRP86 3.89 Å` — the defining donepezil interaction — plus Trp286, Tyr341,
His447 and cation-π to Tyr337. **1HSG** gives `Cationic ASP25 4.00 Å`, the
catalytic dyad. **1M17** is rejected at 2.6 Å and **10CX** is rejected because
its "ligand" is polyethylene glycol.

Label statistics on the first 78 complexes: 30.7% of ligand atoms carry an
interaction, with π-stacking the commonest single type at 14.7%.

### Pipeline, all runnable

```bash
PYTHONPATH=. python3 -m pharmacophore.build   --release corpus/v1 --workers 6
PYTHONPATH=. python3 -m pharmacophore.splits  --release corpus/v1
PYTHONPATH=. python3 -m pharmacophore.dataset --release corpus/v1
PYTHONPATH=. python3 -m pharmacophore.baselines --release corpus/v1
PYTHONPATH=. python3 -m pharmacophore.train   --release corpus/v1 --epochs 30
```

A smoke test of the whole chain on 97 training complexes completed, and the
ablation gate fired correctly: *"removing the pocket costs only 0.0147, below
the 0.02 margin: the model is predicting from the ligand alone."* Expected at
that size — the question is whether it still says that at 17,000 complexes, and
that is the single most important number this project will produce.

### The GPU moment

Not yet. Everything above runs on CPU or MPS. The GPU becomes worth renting when
the corpus is large enough for the training arms to take hours rather than
seconds — roughly once `both_unseen` has a few hundred test complexes, which
needs ~30,000 complexes because that regime is intrinsically ~1% of the corpus.

## Running it

```bash
# one-time, outside the repo (see requirements-corpus.txt for why)
/opt/homebrew/bin/python3.12 -m venv ~/.venvs/smileberta
~/.venvs/smileberta/bin/pip install -r pharmacophore/requirements-corpus.txt

# resume the build — skips everything already recorded
PYTHONPATH=. ~/.venvs/smileberta/bin/python -m pharmacophore.build \
    --release corpus/v1 --workers 4

# the manifest, any time, without disturbing a running build
PYTHONPATH=. ~/.venvs/smileberta/bin/python -m pharmacophore.build \
    --release corpus/v1 --summarise-only

# tests (14, including the known-binding-mode checks)
PYTHONPATH=. ~/.venvs/smileberta/bin/python tests/test_pharmacophore.py
```

Resume is safe at any point: each entry's outcome is appended to a JSONL shard
as it completes and a restart skips what is already recorded. Killing the build
mid-write logs a `BrokenPipeError` from the worker pool; the records on disk are
unaffected.

Structures are **not** cached (`SB_CACHE_STRUCTURES=1` to override). At ~1 MB
each, caching the candidate set would need ~140 GB against 17 GB free. The
labels are kilobytes, so the records persist and a restart re-downloads only
the entry it was working on.

## Two things to fix before the full run

1. **`ligand_incomplete_or_unmatched` is the largest rejection category** — 79 of
   the first 150 entries. That is RDKit failing to match deposited ligand
   coordinates onto the chemical-component definition, via
   `redock.crystal_mol` → `AssignBondOrdersFromTemplate`. Some of those ligands
   are genuinely disordered or truncated and should be dropped, but 79 is high
   enough that a technical matching failure is likely mixed in, and that would
   be discarding valid complexes. Diagnose before spending 14 hours.

2. **The candidate list is alphabetical**, so the first entries are `100D`,
   `101D`, `102L` — disproportionately nucleic-acid and legacy structures, which
   is why 42 of 150 were mmCIF-only. Shuffle the IDs with a fixed seed before
   resuming. Same total work, but partial progress becomes a representative
   sample of the whole PDB, so a half-finished corpus is still usable and its
   statistics mean something at any point.

## Rejection reasons, first 150 entries

Unrepresentative for the reason above, but the shape is informative:

| Reason | Count |
| --- | --- |
| `ligand_incomplete_or_unmatched` | 79 |
| `no_drug_like_ligand` | 75 |
| `no_pdb_format` (mmCIF-only) | 42 |
| `known_artefact` | 33 |
| `acyclic_likely_additive` | 27 |
| `no_uniprot_mapping` | 4 |

## Known biases, recorded rather than hidden

- **Acyclic ligands are excluded.** The artefact filter requires a ring, because
  cryoprotectants and buffers are flexible acyclic chains while essentially
  every drug and fragment has a ring. This also removes genuinely acyclic
  ligands — some peptidomimetics and fatty-acid binders.
- **Cofactors are excluded from the ligand role** (haem, NAD, FAD, ATP and
  relatives) while remaining part of the receptor. They are real binders but
  appear in thousands of unrelated structures and would dominate any
  frequency-based signal.
- **mmCIF-only entries are lost.** `gemmi` is in the requirements to recover
  them later; the rejection count makes the cost visible.
- **Histidine is left neutral.** Its pKa sits near physiological pH and guessing
  wrong in either direction would plant a systematic error in the labels.

## Module map

| File | Does |
| --- | --- |
| `sources.py` | Cached fetching, rate limiting, SIFTS mapping, licences |
| `complexes.py` | Entry and ligand selection, artefact filter, pocket extraction |
| `protonate.py` | pH 7.4 hydrogens with residue numbering preserved |
| `interactions.py` | Per-atom typed interaction fingerprints (ProLIF) |
| `build.py` | Resumable parallel orchestrator and manifest |
| `expand_fragments.py` | BRICS corpus expansion (from the benchmark work) |

`sources.py` is the single place every data source's licence is declared, so
`/credits` and the release manifest generate from it rather than from memory.
