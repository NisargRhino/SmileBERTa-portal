# Paper plan: per-atom pharmacophore prediction at fragment scale

Working notes, written before the final numbers landed so the claims are fixed
in advance of seeing them. Where a number is still open the plan says what each
outcome would mean, so neither result can be written up as if it were the one
we hoped for.

## The claim

Docking is the standard way to decide which fragment binds where, and it is
weakest exactly where fragment-based drug discovery operates. We replace the
label, not the method: instead of scoring a pose, predict for every heavy atom
of a ligand which interactions it makes, supervised by the geometry of
crystallographic complexes.

Why this is worth saying:

1. **The label it replaces is measurably bad at fragment scale.** Our own
   earlier model was trained on Vina scores and `BENCHMARK.md` found 4.5% of its
   outputs were genuine substructures of their own input and a constant
   baseline answering "benzene" beat it. That is a negative result we produced
   and published against ourselves, which makes the motivation concrete rather
   than rhetorical.
2. **The task is defined so the failure mode cannot recur.** The model labels
   atoms of the molecule it is given. A predicted pharmacophore is a
   substructure of the input by construction, so the hallucination rate is zero
   rather than measured.
3. **How the pocket is delivered decides whether it is used at all**, and we
   can show that with three mechanisms on identical data.

## The result the paper turns on

Three ways of giving a model the same pocket, each measured against **its own**
no-pocket control:

| Mechanism | Control |
| --- | --- |
| Random forest, composition as per-atom columns | same forest, columns removed |
| Network, six composition numbers added to every atom alike | same network, `use_pocket=False` |
| Network, each atom attending its own nearest pocket atoms | same network, `use_pocket=False` |

For both networks the control is the same architecture with the pocket masked —
identical parameter counts, depth and schedule — so a difference between arms is
the conditioning and not capacity.

Already measured and replicated across three independent cycles: the forest
gains ~+0.04 macro AUC from the pocket while the descriptor network gains
between −0.013 and +0.003. The information is present and reachable; that
conditioning mechanism cannot deliver it.

**Open at the time of writing:** whether geometric conditioning converts this.

* If it does, the paper is *a conditioning mechanism that fails, a diagnosis of
  why, and one that works*, which is a stronger and less common shape than "our
  model scores well".
* If it does not, the paper is a negative result with unusually tight controls:
  the signal is demonstrably in the data, a weak model extracts it, and two
  neural conditioning schemes do not. That is publishable and useful, and it
  would redirect the next attempt rather than leaving it to guesswork.

Either way the write-up reports both networks. Deleting the failed arm once a
better one exists would be rewriting history.

## Evidence that is already solid

- **48,150 complexes**, 8,121 proteins, 25,042 ligands, 1,195,427 labelled
  heavy atoms, from 138,275 candidate PDB entries.
- **Three held-out regimes** with verified zero scaffold and zero protein
  leakage: unseen ligand, unseen protein, both unseen.
- **Labels validated against textbook binding modes** rather than only in
  aggregate: 3PTB's benzamidine salt bridge to Asp189 at 2.92 Å, donepezil's
  face-to-face stacking with Trp86 in 4EY7, the catalytic dyad in 1HSG, and
  CDK2's hinge motif recovered in 4D1Z without the pipeline being told about
  kinases.
- **A measured ceiling on any ligand-only model**: across 63,218 pairs of the
  same ligand bound to two different proteins, median label overlap is 0.26,
  only 2.1% of pairs are labelled identically and 19.8% share no label at all.
  A ligand-only model must give one answer for every one of those pairs.
- **Reproducibility**: committed splits with digests, licences declared per
  source, figures generated from the results files with the plotted values
  written beside each one.

## Methodological points worth a paragraph each

These were forced by things that went wrong, which is the honest reason to
include them:

- **Macro averages need a support floor.** Averaging an AUC computed on two
  positive atoms equally with one computed on nine thousand produced a sign
  change between consecutive runs of the same measurement. Aggregates now
  require 50 positives; every type is still reported individually.
- **Corpus-wide ablations are diluted.** Only ~1 ligand in 8 binds more than one
  protein; for the rest, memorising the molecule is the correct answer and the
  pocket cannot change it. Reporting the ablation restricted to promiscuous
  ligands raised the measured effect by ~40%.
- **Pocket descriptors can smuggle in ligand size.** The residue count within
  6.5 Å of the ligand correlates with ligand heavy-atom count at r = 0.64. It
  was removed; the effect survived, because the count on its own scored *below*
  the ligand-only arm.

## What the paper cannot claim yet

- **Apo proteins.** Pockets here are defined by a bound ligand. Scoring a
  protein with nothing in it needs a pocket finder first, and the shift from
  crystallographic to predicted pockets is untested. Any "give us a protein"
  claim is out of scope until that is measured.
- **Prospective validation.** Everything is retrospective on held-out
  structures. No prediction has been tested against an experiment.
- **Comparison with published baselines.** Our arms are compared to each other,
  not to DeepICL, MaSIF or the pocket-conditioned generators. Until that is run
  on a shared benchmark, "better" is not available as a word.

## Venue

Not Nature. Per-atom interaction conditioning already exists in the literature —
DeepICL (Nat. Commun. 2024) assigns interaction conditions per protein atom, and
pocket-conditioned diffusion models give each ligand atom its nearest 32 pocket
atoms in 3D. A Nature-family venue would need a methodological advance over
those plus validation we do not have.

Realistic and appropriate: **Journal of Cheminformatics, JCIM, Bioinformatics,
Digital Discovery**. The contribution is the corpus, the evaluation discipline
and the conditioning comparison, not a new state of the art.

The route to a higher-impact venue is the third gap above: beat published
baselines on a shared benchmark, then validate one prediction prospectively.

## Figures

| Figure | Shows |
| --- | --- |
| `corpus_overview` | what the corpus contains and what was rejected |
| `protein_coverage` | concentration, the first bias a reviewer checks |
| `ligand_promiscuity` | the ceiling on any ligand-only model |
| `conditioning_comparison` | the central claim, three mechanisms |
| `subset_ablation` | corpus-wide against promiscuous-ligand |
| `per_type_auc_*` | per interaction type with support, never an average alone |
| `training_curves` | selection on validation only |
