# Task: Pharmacophore Prediction — both directions, one conditioned model

> Requires `PHARMACOPHORE_DATA_TASK.md`. Do not start training until that corpus and its
> three split regimes exist.

## Context

This replaces the fragment prediction tool. `BENCHMARK.md` records why: on an honest
scaffold split the shipped fragment model returns a genuine substructure of its own input
4.5% of the time, emits a parseable SMILES 41% of the time, and loses to a constant
baseline that always answers benzene. Regularisation changed nothing and sixteen times more
data changed nothing, because the labels were docking scores rather than measurements.

The replacement predicts something that was observed: which atoms of a ligand interact with
a protein, and which interaction types a pocket demands.

## What we are building

Two user-facing capabilities, which are **one model with two conditionings**, not two
systems:

1. **Given a molecule and a target — which part of it is the pharmacophore?**
   Per-atom interaction prediction. The answer is a substructure of the input by
   construction, so it cannot hallucinate a fragment from another molecule. This is the
   honest version of what the fragment tool was trying to do.

2. **Given a protein pocket — what should bind here?**
   Predict the pocket's required pharmacophore feature pattern, then retrieve and rank real
   fragments whose feature profiles match. Works on a protein never seen before, which is
   the capability no amount of per-target training gives you.

Training one model across all targets and conditioning on the pocket gives both. The
alternative — a separate model per protein — needs a large dataset for every target
separately, and would make KRAS a dead end rather than a first example.

## The thing that will break this if done naively

**Per-atom labels are heavily imbalanced and trivially gamed.** Most ligand atoms make no
notable interaction. A model that predicts "no interaction" everywhere scores well on
accuracy and is worthless. Report per-interaction-type precision, recall and AUC with
support attached, exactly as the classification benchmark does per class, and never lead
with a single accuracy number.

The second trap is subtler. **A model can learn the ligand and ignore the pocket entirely.**
Most ligands in the corpus appear in one complex, so memorising "this molecule's atoms 4, 7
and 12 interact" reproduces the training labels without any protein understanding at all.
The ablation in §4 is not optional — it is how you find out whether you built what you
think you built.

---

## 1. Baselines, before any neural network

`BENCHMARK.md` found a random forest on ECFP4 beating a 44M-parameter transformer by 0.33
macro-F1. That result is why this section exists and why it comes first.

**For per-atom prediction:**
- Atom-type frequency prior: for each element and hybridisation, the base rate of each
  interaction type across the training set. A surprising amount of pharmacophore signal is
  "carboxylates make salt bridges".
- Random forest on atom-centred ECFP-style environment features, no pocket information.
  **This is the ablation baseline** — whatever the full model achieves, it must beat the
  version that cannot see the protein, or it has not learned about proteins.
- Buried-surface heuristic: predict interaction for atoms a simple geometric calculation
  says are buried.

**For pocket → fragments:**
- Global fragment frequency: rank by how often each fragment appears in any complex. The
  equivalent of the benzene baseline, and it will be harder to beat than it sounds.
- Nearest-pocket retrieval: find the most similar pocket in the training set by descriptor
  distance and return its ligands' fragments. If the learned model cannot beat this, it is
  an expensive lookup table.

Publish all of them alongside the model, in the same table, on all three split regimes.

## 2. Models

**Ligand encoder:** ChemBERTa-2, or whatever is current at build time — compare at least two
and record why one was chosen. A graph neural network over the molecular graph is a
legitimate alternative and may suit per-atom prediction better, since the task is node
classification and SMILES token positions map awkwardly onto atoms. Try both.

**Pocket encoder:** ChemBERTa speaks only SMILES and knows nothing about proteins. Options:
- ESM-2 embeddings of the pocket residues — simple, sequence-only, ignores geometry
- a geometric GNN over pocket atoms in 3D — more faithful, since pharmacophores are
  geometric arrangements and not residue lists

Start with ESM-2 for a working baseline, then test whether geometry earns its complexity.

**Joining them:** cross-attention from ligand atoms to pocket residues, with a per-atom head
emitting one probability per interaction type. The attention map is a free interpretability
artefact — which pocket residue the model thinks each ligand atom is talking to — and that
is worth surfacing in the UI.

**Direction 2** reuses the same trained encoders: run the pocket encoder alone to predict
the feature pattern the pocket requires, then score a fragment vocabulary against it. The
vocabulary is built by BRICS/RECAP decomposition over ChEMBL and filtered to purchasable
compounds via `BUYABILITY_TASK.md`, so every suggestion is a fragment someone can actually
order.

## 3. Metrics

**Per-atom (direction 1):** per-interaction-type precision, recall, F1, AUC and support.
Macro and micro averages. Atom-level agreement with the held-out crystal contacts.
Additionally: does the predicted pharmacophore, taken as the connected substructure of
high-scoring atoms, match the fragment a chemist would identify? Compare against the
BRICS/RECAP decomposition as a sanity check, not as truth.

**Retrieval (direction 2):** recall@k of the fragments actually observed bound in held-out
complexes. Report recall@10, @50 and @100 with the vocabulary size stated, since recall
without a denominator is meaningless.

**Every metric, on all three split regimes, always.** The unseen-protein and both-unseen
columns are the ones that matter and they will be the worst ones.

## 4. Required ablations

Run these and publish them. They are how the claims stay honest:

- **No pocket.** Ligand encoder only. If performance barely drops, the model is not using
  the protein and direction 2 does not work, whatever the headline number says.
- **Shuffled pockets.** Pair each ligand with a random pocket. Performance should collapse.
  If it does not, something is leaking.
- **Pocket only.** No ligand information, for direction 2's realistic setting.
- **Sequence-only versus 3D pocket encoder**, to establish whether geometry pays for itself.

## 5. KRAS, and the per-target claim

KRAS is the first demo target. It carries **17,042 ChEMBL activity records (5,903 IC50, 348
Kd)** and several hundred PDB entries, so it is well enough covered to be a real example.

Train the general model first. Then fine-tune on KRAS and compare on the honest split.

**Ship the per-target model only if it beats the general model on held-out data.** This is
the rule `MODEL_HONEST_EVAL_TASK.md` §6 already sets for the existing target selector:
*"If a target does not have enough data to beat the shared model on the honest split, ship
the shared model for that target and say so in the UI."* The same rule applies here, and it
governs what the website may claim. "We can train a model for your protein" is only an
honest offer if the per-target numbers justify it; otherwise the offer is to run the
general model, and the site should say so.

## 6. Website

Replace the fragment prediction tool. The target selector becomes functional — it currently
records a choice that changes nothing, because one checkpoint answers for every target.

- Molecule in, target selected: the molecule rendered with per-atom interaction confidence,
  the predicted pharmacophore highlighted, and the pocket residues each key atom is
  predicted to contact.
- Pocket in (PDB ID, or a structure from the docking tool): ranked fragment suggestions,
  each with its predicted interaction pattern, its buyability badge, and a link to dock it.
- Every result links to the benchmark numbers for the regime it falls in — a prediction for
  a protein family absent from training must say so on the result, not in a footnote.
- Keep the honest caveat pattern the ADMET and docking pages already use: these are ranked
  hypotheses from structural precedent, not binding predictions.

## 7. Publication

The evaluation is the contribution. Pocket-conditioned ligand design has substantial prior
art — Pocket2Mol, TargetDiff, DiffSBDD among others — and a new architecture is not what
this project has to offer. What is genuinely scarce is a system that publishes its
recall on a both-unseen split alongside the baselines it beats and the ablations that show
where its performance comes from.

`BENCHMARK.md` is already part of that story: a measured negative result on the previous
approach, with the contamination signature that explains the old numbers. Few groups
publish that, and it is what makes the new numbers credible.

---

## Tests

- A predicted pharmacophore is always a connected substructure of the input molecule.
- The no-pocket ablation runs and is reported; a run that omits it fails CI.
- Shuffled-pocket performance is materially worse than true-pocket performance; if it is
  not, the build fails, because that means a leak.
- Retrieval returns only fragments from the declared vocabulary, never a generated string.
- A protein with no homologue in training is flagged as out-of-distribution in the response.
- KRAS fine-tuning is compared against the general model on held-out data, and the UI claim
  follows the comparison rather than the intent.
- Known pharmacophores are recovered: for a handful of well-characterised complexes, the
  predicted key atoms overlap the published binding determinants.

## Do not

- Do not generate SMILES. Direction 2 retrieves from a vocabulary; that is what makes every
  suggestion real and keeps the hallucination rate at zero by construction.
- Do not train on a docking score, anywhere, for anything.
- Do not report a single accuracy number for per-atom prediction. Most atoms are negatives
  and the number would be meaningless.
- Do not ship a per-target model that loses to the general model, however much more
  specific it sounds.
- Do not describe outputs as binding predictions or affinities. They are structural
  hypotheses with a measured recall.
- Do not let the ligand encoder see pocket-derived features that were computed from the
  bound complex. That is the leak that makes an ablation look healthy when it is not.
