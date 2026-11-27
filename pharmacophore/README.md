# Pharmacophore corpus

Protein–ligand complexes annotated with the interactions each ligand atom
actually makes, built to replace docking scores as the fragment model's
supervision. See `PHARMACOPHORE_DATA_TASK.md` for the spec and
`BENCHMARK.md` for why the old approach is being replaced.

## State as of 2026-11-26

**Pipeline built and validated. Corpus build paused at entry 150 of 139,597.**

| | |
| --- | --- |
| Candidate entries | 139,597 (X-ray, ≤ 2.5 Å, has a non-polymer entity) |
| Processed | 150 |
| Complexes kept | 23 |
| Records | `corpus/v1/records/*.jsonl` (gitignored, ~1 MB so far) |
| Candidate ID list | `corpus/v1/candidate_ids.json` — fetched once, reused |
| Estimated full run | ~14 h at 170 entries/min with 4 workers |

Validated against known binding modes: **3PTB** gives Asp189 at 2.92 Å and
Gly219 at 2.82 Å, **1HSG** gives the catalytic Asp25 at 2.77 Å, **4EY7** gives
Trp86 / Trp286 / Tyr341. **1M17** is correctly rejected at 2.6 Å, and
**10CX** is correctly rejected because its "ligand" is polyethylene glycol.

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
