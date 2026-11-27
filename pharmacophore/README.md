# Pharmacophore corpus

Protein–ligand complexes annotated with the interactions each ligand atom
actually makes, built to replace docking scores as the fragment model's
supervision. See `PHARMACOPHORE_DATA_TASK.md` for the spec and
`BENCHMARK.md` for why the old approach is being replaced.

## State as of 2026-11-27, 10:30

**Full pipeline built; corpus build 44% done and still running** under a watchdog
that has not had to intervene — the health log shows an unbroken five-minute
cadence from 00:20 to 10:31 with 7 processes alive throughout.

| | |
| --- | --- |
| Candidate entries | 139,597 (X-ray, ≤ 2.5 Å, has a non-polymer entity) |
| Entries processed | 61,078 (44%) |
| **Complexes kept** | **20,982** |
| Unique proteins / ligands | 5,102 / 12,709 |
| Labelled heavy atoms | 521,265, of which 30.7% carry a contact |
| Build rate | ~115 entries/min, 6 concurrent batches |
| Records | `corpus/v1/records/*.jsonl` (gitignored), one line per complex |
| Health log | `corpus/v1/health.log` — a snapshot every 5 minutes |

All nine interaction types are populated, which was not true before the geometric
detectors were added:

| Type | Atoms | | Type | Atoms |
| --- | --- | --- | --- | --- |
| Hydrophobic | 42,208 | | Cationic | 7,000 |
| HBAcceptor | 38,557 | | PiCation | 4,067 |
| Anionic | 22,890 | | XBDonor | 636 |
| HBDonor | 21,343 | | MetalDonor | 67 |
| PiStacking | 19,201 | | | |

Validated against known binding modes. **3PTB** gives `Cationic ASP189 2.92 Å`,
the benzamidine salt bridge. **4EY7** gives face-to-face π-stacking with
`TRP86 3.89 Å` — the defining donepezil interaction — plus Trp286, Tyr341,
His447 and cation-π to Tyr337. **1HSG** gives `Cationic ASP25 4.00 Å`, the
catalytic dyad. **1M17** is rejected at 2.6 Å and **10CX** is rejected because
its "ligand" is polyethylene glycol. Spot-checking the corpus at scale, **4D1Z**
(CDK2) gives face-to-face stacking on Phe82 and an H-bond to Leu83 — the CDK2
hinge motif — with nothing in the pipeline told about kinases.

### Splits at this scale, zero leaks

| Regime | train | validation | test | discarded |
| --- | --- | --- | --- | --- |
| `unseen_ligand` | 13,956 | 1,744 | 1,745 | — |
| `unseen_protein` | 13,956 | 1,744 | 1,745 | — |
| `both_unseen` | 10,803 | 84 | **77** | 6,481 |

`shared_scaffolds: 0` and `shared_proteins: 0` on all three. `both_unseen`
remains the thin regime: it discards every complex straddling the boundary, so
77 test complexes will show a direction but not a tight interval.

### Pipeline, all runnable

```bash
PYTHONPATH=. python3 -m pharmacophore.build   --release corpus/v1 --workers 6
PYTHONPATH=. python3 -m pharmacophore.splits  --release corpus/v1
PYTHONPATH=. python3 -m pharmacophore.dataset --release corpus/v1
PYTHONPATH=. python3 -m pharmacophore.baselines --release corpus/v1
PYTHONPATH=. python3 -m pharmacophore.train   --release corpus/v1 --epochs 30
PYTHONPATH=. python3 -m pharmacophore.figures --release corpus/v1
```

`corpus/overnight.sh` runs that chain on a loop while the build continues,
committing each cycle's splits, results and figures, so an interrupted night
still leaves a complete and self-consistent set of numbers.

### Does the pocket contribute? The ablation at 21,000 complexes

At 97 complexes the gate said the pocket was being ignored. At scale the random
forest disagrees in all three regimes, most strongly in the one that
memorisation cannot explain:

| Regime | prior | RF ligand | RF +pocket | delta | verdict |
| --- | --- | --- | --- | --- | --- |
| `unseen_ligand` | 0.5571 | 0.6249 | 0.6794 | **+0.0545** | uses the pocket |
| `unseen_protein` | 0.6400 | 0.6676 | 0.7034 | **+0.0358** | uses the pocket |
| `both_unseen` | 0.5561 | 0.6066 | 0.6623 | **+0.0557** | uses the pocket |

**The transformer does not agree, and that is the open question.** On
`unseen_ligand` it reaches 0.7973 with the pocket and 0.7911 without — a delta
of 0.0070, below the margin, so its gate still reads "predicting from the ligand
alone" even though the forest's reads the opposite on the same split.

Two readings, with different fixes, which is why neither has been applied yet:

1. The transformer's ligand-only arm is already far stronger than the forest's
   (0.79 against 0.62). Self-attention extracts most of what the molecule alone
   can say, so the pocket's *marginal* contribution is genuinely small even
   though the pocket carries real information — which is what the forest, a
   weaker ligand-only model, is able to show.
2. The conditioning is too weak. The pocket vector is projected and **added
   identically to every atom**, so it can shift all atoms together but cannot
   express that *this* atom faces *that* residue. Cross-attention from atoms to
   pocket residues would, at the cost of making the no-pocket ablation a
   different architecture rather than one flag.

Note also that `unseen_protein` has the *highest* frequency prior at 0.6400.
Held-out proteins are easier for a context-free baseline than held-out
chemistries, because the per-atom label distribution is more stable across
proteins than across scaffolds. Worth stating in the paper.

### The GPU moment

Everything above runs on CPU or MPS; an epoch on 13,956 training complexes takes
about 20 seconds on MPS, so the bottleneck is still the corpus, not the compute.
The GPU becomes worth renting once `both_unseen` has a few hundred test
complexes — that regime is intrinsically ~1% of the corpus and currently has 77,
so it needs the build to finish.

## Running it

The interpreter must live **outside the repo** — the repo is in iCloud Drive, where
`import rdkit` can block for minutes on evicted files (see `requirements-corpus.txt`).

```bash
# one-time, anywhere outside the repo
/opt/homebrew/bin/python3.12 -m venv ~/.venvs/smileberta
~/.venvs/smileberta/bin/pip install -r pharmacophore/requirements-corpus.txt
export SB_PY=~/.venvs/smileberta/bin/python

# resume the build — skips everything already recorded
PYTHONPATH=. $SB_PY -m pharmacophore.build --release corpus/v1 --workers 6

# the manifest, any time, without disturbing a running build
PYTHONPATH=. $SB_PY -m pharmacophore.build --release corpus/v1 --summarise-only

# tests (14, including the known-binding-mode checks)
PYTHONPATH=. $SB_PY tests/test_pharmacophore.py
```

**The environment that produced the current run is not at that path.** It was built
in a session scratchpad under `/private/tmp`, because disk was at 98% and a second
copy of torch plus rdkit needs ~3 GB. A reboot or a tmp sweep will delete it, and
the corpus records are unaffected but nothing can be trained until it is rebuilt
with the commands above. Rebuild it to `~/.venvs/smileberta` once there is room;
that is the only reason the two paths differ.

Resume is safe at any point: each entry's outcome is appended to a JSONL shard
as it completes and a restart skips what is already recorded. Killing the build
mid-write logs a `BrokenPipeError` from the worker pool; the records on disk are
unaffected.

Structures are **not** cached (`SB_CACHE_STRUCTURES=1` to override). At ~1 MB
each, caching the candidate set would need ~140 GB against 17 GB free. The
labels are kilobytes, so the records persist and a restart re-downloads only
the entry it was working on.

## The two pre-run fixes, both resolved

1. **`ligand_incomplete_or_unmatched` looked like a matching bug** — 79 of the
   first 150 entries, high enough that RDKit might have been discarding valid
   complexes. At scale it is **1,270 of 44,134 rejections, under 3%**. The spike
   was the alphabetical ordering concentrating legacy and nucleic-acid entries,
   not a defect in `AssignBondOrdersFromTemplate`. Nothing to fix.

2. **The candidate list was alphabetical**, so progress was unrepresentative and
   a half-finished corpus meant nothing. IDs are now shuffled with a fixed seed
   (`--shuffle-seed`), so the statistics above are an unbiased sample of the
   whole PDB and stay valid at any percentage complete.

## Rejection reasons, 61,078 entries

| Reason | Count |
| --- | --- |
| `no_drug_like_ligand` | 29,596 |
| `acyclic_likely_additive` | 3,882 |
| `no_pdb_format` (mmCIF-only) | 3,765 |
| `known_artefact` | 3,512 |
| `ligand_incomplete_or_unmatched` | 1,270 |
| `no_uniprot_mapping` | 802 |
| `no_interactions_detected` | 737 |
| `interaction_typing_failed` | 570 |

`no_drug_like_ligand` dominating is expected and correct: most PDB entries with a
non-polymer entity carry only ions, waters or buffer components.
## Two bugs found at scale that small runs hid

**The resolution gate was passing everything.** `REMARK 2` reads
`REMARK   2 RESOLUTION.    2.00 ANGSTROMS.`, and splitting the whole line finds
the remark number `2` before the actual figure, so every structure in the PDB
came back as exactly 2.0 Å and the ≤ 2.5 Å limit never rejected anything. It
surfaced only because a figure of the resolution distribution was a single bar.
The corpus is **not** contaminated — all 21,401 kept complexes check out against
the wwPDB bulk index, because the candidate query already filters on resolution
at the RCSB end, so the gate was redundant as well as broken. Resolution now
comes from `sources.resolution_index()`, one cached download of `resolu.idx`
rather than a hundred thousand entry-API calls, and the figures prefer it over
the stored value so existing records are repaired for reporting without a
rebuild. True median is 1.9 Å.

**RDKit's recursion was killing the interpreter.** `MolOps::getMolFrags` walks
the molecular graph recursively and ProLIF calls it when converting a receptor;
on a large protein that runs off the 8 MB main-thread stack and the process dies
with `SIGBUS`, which no Python handler can intercept. It cost a whole batch
every few minutes. A thread's stack size is settable where the main thread's is
not, so `run_batch` now runs the work on a thread with a 512 MB stack
(`SB_STACK_BYTES`). Entries are also logged to `corpus/v1/attempts/` *before*
their work begins, so an entry that kills the interpreter twice is recorded as
`crashed_during_processing` instead of being retried forever.

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
| `splits.py` | Three held-out regimes with a leak check |
| `dataset.py` | Per-atom features, pocket descriptors, label matrices |
| `metrics.py` | Per-type AUC/AP/P/R/F1 and the ablation verdict |
| `baselines.py` | Frequency prior and random forests, with and without pocket |
| `model.py` | Pocket-conditioned set transformer over ligand atoms |
| `train.py` | Training with the three arms wired as gates |
| `figures.py` | Figures and analytics; each PNG beside the JSON it plots |
| `expand_fragments.py` | BRICS corpus expansion (from the benchmark work) |

`sources.py` is the single place every data source's licence is declared, so
`/credits` and the release manifest generate from it rather than from memory.
