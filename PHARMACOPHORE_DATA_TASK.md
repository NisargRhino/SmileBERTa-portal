# Task: The Pharmacophore Corpus — one cited dataset, built from experiments

> Build this before `PHARMACOPHORE_MODEL_TASK.md`. Both models in that spec read from this
> corpus, and neither is worth training until it exists.

## Context

Repo: `SmileBERTa-portal`. Flask + Python 3.11 API, RDKit server-side, Neon Postgres, a
dedicated 2-core docking host with AutoDock Vina 1.2.7 and Open Babel, vanilla JS frontend.

The fragment model currently shipped was trained on 219 molecules whose "correct" fragment
was chosen by docking score. `BENCHMARK.md` measures what that produced: 4.5% of its
outputs are genuine substructures of their own input, 41% parse as SMILES at all, and a
constant baseline that always answers benzene beats it on every metric. Sixteen times more
data from the same pipeline moved nothing, and no regularisation intervention helped.

That is a label problem, not a tuning problem, and it is why this corpus exists.

## What we are building

A versioned, cited dataset of **protein–ligand complexes annotated with the interactions
each ligand atom actually makes**, joined to measured bioactivity where it exists.

The central change from what came before: every label is read off an experiment. A
crystal structure records where a ligand sat and which protein atoms it touched. That is a
measurement. A docking score is a prediction from a function with roughly ±2 kcal/mol of
error, and at fragment size that error exceeds the entire range being ranked.

## The thing that will break this if done naively

**Splitting by ligand alone leaks through the protein, and splitting by protein alone leaks
through the ligand.**

The same ligand appears bound to many homologous proteins; the same protein appears bound
to hundreds of analogues. A model tested on an unseen ligand against a protein it has seen
a thousand times has not been tested on anything hard. A model tested on an unseen protein
with a ligand series it memorised is just as compromised.

Three held-out regimes must be constructed and reported separately, always:

| Regime | Held out | Answers |
| --- | --- | --- |
| **Unseen ligand** | ligand scaffold | Can it generalise chemistry on a known target? |
| **Unseen protein** | protein cluster at 30% sequence identity | Can it handle a target it has never seen? |
| **Both unseen** | both | The honest number, and the one to publish |

Expect "both unseen" to be dramatically worse than the other two. That gap is a finding,
not a failure, and it is the single most informative number this project will produce.

---

## 1. Sources, and their licences

**Verify every licence before anything derived from a source is redistributed or shipped.**
The platform's `/credits` page gains an entry per source with its citation and terms. Part
of what is being replaced here is a model trained on "scattered drugs from three databases
that aren't quoted".

| Source | What for | Licence — confirm before use |
| --- | --- | --- |
| RCSB PDB | complexes, poses, contacts | public domain (CC0) |
| ChEMBL | measured activity, ligand efficiency | CC BY-SA 3.0 — share-alike propagates |
| BindingDB | additional affinity | CC BY 3.0 |
| PubChem | identifiers, cross-references | public domain |
| PDBbind / CASF | curated affinity, standard benchmark | academic licence; check redistribution |
| Fragalysis / XChem | crystallographic fragment screens | open; confirm per-campaign terms |
| ZINC in-stock | buyability of suggested fragments | see `BUYABILITY_TASK.md`; check terms |

Measured at the time of writing: **142,213 PDB entries hold at least one non-polymer entity
at ≤ 2.5 Å resolution**, and ChEMBL holds **2,921,148 molecules**. These are the raw pools;
the sections below cut them down considerably, and the real sizes must be recorded rather
than estimated.

## 2. Selecting complexes

Reuse `redock.py` — its ligand detection is already written and already tested.

Keep a complex when all hold:

- X-ray, resolution ≤ 2.5 Å, with ligand occupancy and B-factors recorded for later filtering
- the ligand is a single chemical component of **≥ 8 heavy atoms**, non-covalent, and not in
  the solvent/buffer/cryoprotectant exclusion list in `redock.EXCLUDE_RESIDUES`.
  *(Corrected from 12 during the build: 12 is the docking panel's drug-like floor, but a
  corpus built to learn which fragments bind a pocket must not exclude fragments. Fragment
  screens deposit ligands of 7–15 heavy atoms and 3PTB's benzamidine has 9. Heavy-atom
  count is recorded per record, so a drug-like-only subset remains one filter away.)*
- cofactors and prosthetic groups (haem, NAD, FAD, ATP and relatives) are excluded from the
  **ligand** role — they are real binders but appear in thousands of unrelated structures
  and would dominate any frequency signal — while remaining part of the receptor
- the ligand is complete: no missing heavy atoms against its chemical-component definition
- the protein chain maps to a UniProt accession

Record every rejection with its reason, and publish the counts. A corpus whose exclusions
are invisible cannot be audited, and `REDOCK_VALIDATION_TASK.md` established that the
rejection log is part of the product.

**Covalent ligands are excluded, not silently included.** `redock.py` already detects them
by heavy-atom distance; a covalent adduct's contact pattern is not comparable with
non-covalent binding and mixing them teaches the model a chimera.

## 3. The labels: per-atom interaction fingerprints

For every kept complex, compute for each **ligand heavy atom** the interactions it makes:

- hydrogen bond (donor / acceptor, with geometry)
- hydrophobic contact
- π-stacking (face-to-face, edge-to-face) and π–cation
- salt bridge, with formal charges
- halogen bond
- metal coordination
- solvent-accessible surface area buried on binding

Use **ProLIF** (RDKit-native) or **PLIP**, both open source, rather than hand-rolling the
geometry. `docking.py:analyze_contacts` already computes the 4 Å contact and H-bond layer
and is the fallback if a dependency cannot be added to the API tier.

This is what makes the output a *pharmacophore* rather than a bag of atoms: a typed
interaction pattern with geometry, which is what a medicinal chemist means by the word.

Store per complex: the per-atom typed interaction vector, the pocket residues, the ligand's
SMILES and its canonical form, and the buried-surface fraction.

## 4. The pocket representation

A pocket is the protein residues with any heavy atom within 6.5 Å of any ligand heavy atom,
plus their sequence neighbours for context. For each pocket record:

- residue identities and their 3D coordinates
- a pharmacophore-feature map of the pocket itself — which feature types the pocket can
  *satisfy*, and where. This is the target side of the same alphabet used for the ligand,
  which is what lets one model speak about both.
- volume, buried-surface area, hydrophobicity, net charge, and H-bond donor/acceptor counts
- the UniProt accession, gene symbol, and protein family

**The pocket representation must not include the ligand.** It is trivially easy to leak
ligand information into "pocket" features computed from a holo structure — pocket volume
measured around a bound ligand, for instance, encodes the ligand's size. Every pocket
descriptor has to be computable for an apo structure or a structure whose ligand has been
stripped, and there should be a test that asserts it.

## 5. Bioactivity join

Join ChEMBL and BindingDB activities to complexes by UniProt accession and ligand InChIKey.
Keep `pChEMBL`, assay type, and confidence score; prefer `Kd` and `Ki` over `IC50` and
record which was used. Compute **ligand efficiency** (−ΔG per heavy atom), since that is
how fragment binding is actually judged and raw potency favours large molecules.

Measured for the first demo target: **KRAS (CHEMBL2189121) carries 17,042 activity records
— 5,903 IC50, 348 Kd, 43 Ki.** That is enough for a per-target model, though deduplication
and confidence filtering will cut it. Record what survives.

Most complexes will have no activity record. That is fine: the structural labels stand on
their own, and the activity join is an extra supervision signal where it exists, not a
requirement for inclusion.

## 6. Splits

Build all three regimes from §"The thing that will break this", using the machinery in
`evaluation/splits.py`, which already assigns whole scaffold groups and verifies that none
straddle the boundary.

- **Ligand axis**: Bemis–Murcko scaffold groups, as the existing code does.
- **Protein axis**: cluster sequences at 30% identity (MMseqs2 or CD-HIT) and keep whole
  clusters together. Splitting by UniProt accession alone is not enough — close homologues
  carry the same pocket.

Write every split to a committed file with a digest, exactly as `evaluation/splits/` does.
A split regenerated with a different seed makes every number incomparable with every other,
which is the failure the existing split files exist to prevent.

## 7. Release format

One versioned release directory with a manifest: source versions and download dates, the
filter counts at every stage, per-split sizes, sha256 for every file, and the commit that
built it. The corpus is a citable artefact or it is not worth building.

Store the heavy data outside git — it will be gigabytes. Commit the manifest, the splits
and the build scripts.

---

## Tests

- A ligand with a missing heavy atom relative to its component definition is rejected, with
  the reason recorded.
- A covalent ligand is rejected; reuse the 6LU7 case already in `tests/test_redock.py`.
- No protein cluster appears on both sides of the protein-axis split; no ligand scaffold
  appears on both sides of the ligand-axis split; "both unseen" violates neither.
- Pocket descriptors computed with the ligand present equal those computed with it stripped.
  This is the leak test and it is the most important one here.
- Interaction fingerprints for a known complex match published contacts — 3PTB's benzamidine
  salt bridge to Asp189 is the canonical case and is already referenced in the codebase.
- The manifest's digests match the files it describes.
- Rebuilding the corpus from the same source versions reproduces identical splits.

## Do not

- Do not use a docking score as a label anywhere in this corpus. That is the mistake being
  corrected; docking may filter or rank candidates, never define truth.
- Do not include covalent complexes in the main corpus.
- Do not split by ligand alone or protein alone and report it as the headline.
- Do not drop rare protein families to make aggregate numbers look better, for the same
  reason `MODEL_HONEST_EVAL_TASK.md` forbids dropping rare classes.
- Do not redistribute any source whose licence has not been checked and recorded in
  `/credits`.
- Do not train anything in this task. This spec produces a dataset and its manifest, and
  nothing else.
