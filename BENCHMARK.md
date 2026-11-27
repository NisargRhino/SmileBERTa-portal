# Benchmark: scaffold-split evaluation of both SmileBERTa models

Generated 2026-11-26 from commit `b1f718f79a92` by the scripts in `evaluation/`. Every number below is written to `evaluation/results/` by a named run and can be regenerated with the commands in [Reproducing this](#reproducing-this).

> "No published benchmark" was the first item under limitations in the platform overview. This page is that benchmark. It is deliberately unflattering in places: a measurement that only reports good news is not a measurement.

## What this found

- **A random split overstates classification performance by 0.242 macro-F1.** The same random forest scores 0.800 on a random split and 0.558 when no scaffold is allowed to appear on both sides.
- **The shipped classifier shows almost no scaffold gap (0.903 → 0.899), which is a symptom rather than a strength.** An honestly trained model loses a quarter of its macro-F1 on this split. A model that does not was trained on the molecules being used to test it.
- **Only 0.045 of predicted fragments are genuine substructures of the molecule they came from**, and 0.409 of raw outputs parse as SMILES at all. The repair path fires on the rest and returns a fragment of a different molecule.
- **The fragment model is beaten by answering "benzene" every time.** That constant baseline scores 0.500 exact match and 0.955 substructure rate, because the commonest BRICS fragment really is present in most drug-like molecules.
- **A fingerprint model beats the transformer retrained here** (0.558 vs 0.229 macro-F1 on the scaffold split). Read that as a bound on **this training recipe**, not on the architecture: the same recipe reaches only 0.487 even when the test set is included in its training data, well short of the shipped checkpoint, so it is materially weaker than whatever trained that model. What the comparison does establish is that a random forest on fingerprints is a serious baseline here, and any future model has to clear it.
- **Sixteen times more fragment training data changed nothing.** With 20,000 training pairs rebuilt by the same BRICS pipeline, against 1,223 before, validity, exact match and substructure rate are all still 0.000. Combined with the regularisation interventions doing nothing, this points away from both tuning and data volume, and toward the labels themselves: a fragment chosen by docking score is not a measurement, and a model cannot learn a better signal than its labels carry.

## Method

### The datasets

| Dataset | Size | Scaffolds | Notes |
| --- | --- | --- | --- |
| Drug classification | 8,336 compounds | 2,530 | 12 categories; 1,350 carry more than one; 543 have no ring system |
| Fragment prediction | 219 molecules / 1,223 fragments | 173 | BRICS decompositions, 5.58 fragments per molecule on average |

Class sizes across the whole classification dataset:

| Class | Compounds |
| --- | --- |
| Anti-infective | 2,923 |
| Antineoplastic | 1,642 |
| CNS | 1,528 |
| Cardiovascular | 1,262 |
| Anti-inflammatory | 591 |
| Gastrointestinal | 490 |
| Hematologic | 462 |
| Respiratory | 281 |
| Dermatologic | 226 |
| Reproductive | 225 |
| Lipid-regulating | 203 |
| Urological | 48 |

### The scaffold split

Every compound gets its Bemis–Murcko scaffold via RDKit's `MurckoScaffold.MurckoScaffoldSmiles`. Compounds are grouped by scaffold and whole groups are assigned to train, validation or test at 80/10/10, largest group first so the split sizes land close to target. A group is placed in whichever split is furthest below its quota **for the classes that group carries**, so the rare classes survive; perfect stratification is impossible once whole groups must move together, and the real per-class counts are published below rather than smoothed.

Molecules with no ring system have no Murcko scaffold. There are 543 of them, and treating them as a single group would force 6.5% of the dataset into one split, so each is its own group.

The assignment is committed to `evaluation/splits/*.csv` and loaded from there by every script. A split regenerated on the fly with a different seed would make every number in this repository incomparable with every other.

| Dataset · split | Train | Validation | Test | Scaffolds in both train and test |
| --- | --- | --- | --- | --- |
| classification · scaffold | 6652 | 844 | 840 | 0 |
| classification · random | 6669 | 834 | 833 | 382 |
| fragment · scaffold | 175 | 22 | 22 | 0 |
| fragment · random | 175 | 22 | 22 | 4 |

The random split shares hundreds of scaffolds across the train/test boundary. That is not a bug in the random split — it is what a random split *is*, and it is the reason its numbers look so much better.

Per-class counts in the scaffold split:

| Class | Train | Validation | Test |
| --- | --- | --- | --- |
| Anti-infective | 2339 | 292 | 292 |
| Antineoplastic | 1314 | 164 | 164 |
| CNS | 1222 | 153 | 153 |
| Cardiovascular | 1010 | 126 | 126 |
| Anti-inflammatory | 456 | 78 | 57 |
| Gastrointestinal | 392 | 49 | 49 |
| Hematologic | 369 | 47 | 46 |
| Respiratory | 225 | 28 | 28 |
| Dermatologic | 180 | 23 | 23 |
| Reproductive | 179 | 23 | 23 |
| Lipid-regulating | 161 | 21 | 21 |
| Urological | 35 | 6 | 7 |

### A word on the shipped checkpoints

`NisargRhino/drug-classification` and `NisargRhino/protein-models` were trained before this split existed, on an unknown division of the same data. Their training set therefore almost certainly includes molecules in this test set. Their numbers are reported because "what do users get today" is a real question, but they are marked **contaminated** everywhere and they are not the headline. The honest numbers come from models retrained here on the scaffold-split training set alone.

## Drug classification

| Model | Macro-F1 | Micro-F1 | Top-3 | Exact set | Hamming loss |
| --- | --- | --- | --- | --- | --- |
| Majority class | 0.043 | 0.319 | 0.665 | 0.286 | 0.123 |
| Random forest (ECFP4) | 0.558 | 0.633 | 0.879 | 0.534 | 0.069 |
| Logistic regression (ECFP4) | 0.524 | 0.574 | 0.792 | 0.396 | 0.092 |
| **Transformer, retrained on this split** | 0.229 | 0.348 | 0.643 | 0.046 | 0.194 |
| Shipped transformer *(contaminated)* | 0.899 | 0.929 | 0.977 | 0.909 | 0.014 |

All on the scaffold-split test set. *Exact set* is the fraction of compounds where the predicted set of categories matched the true set exactly — the strict multi-label criterion.

### Per class

Support is the number of test compounds carrying that class. Urological is 48 compounds in the entire dataset and is reported here rather than dropped, because dropping it would improve macro-F1 while making the benchmark less true.

| Class | Support | Random forest F1 | Retrained transformer F1 | Shipped *(contaminated)* F1 |
| --- | --- | --- | --- | --- |
| Anti-infective | 292 | 0.767 | 0.590 | 0.973 |
| Antineoplastic | 164 | 0.544 | 0.382 | 0.928 |
| CNS | 153 | 0.658 | 0.173 | 0.957 |
| Cardiovascular | 126 | 0.502 | 0.093 | 0.900 |
| Anti-inflammatory | 57 | 0.741 | 0.130 | 0.860 |
| Gastrointestinal | 49 | 0.340 | 0.217 | 1.000 |
| Hematologic | 46 | 0.339 | 0.222 | 0.771 |
| Respiratory | 28 | 0.488 | 0.170 | 0.619 |
| Dermatologic | 23 | 0.629 | 0.040 | 0.857 |
| Reproductive | 23 | 0.648 | 0.391 | 0.979 |
| Lipid-regulating | 21 | 0.583 | 0.343 | 0.950 |
| Urological | 7 | 0.462 | 0.000 | 1.000 |

On the 133 multi-label compounds in the test set, the shipped model gets the exact set right 0.654 of the time and names at least one correct category 0.977 of the time.

## Random split vs scaffold split

This is the measurement the whole exercise turns on.

| Model | Random split | Scaffold split | Gap |
| --- | --- | --- | --- |
| Random forest (ECFP4) | 0.800 | 0.558 | 0.242 |
| Logistic regression (ECFP4) | 0.763 | 0.524 | 0.239 |
| Transformer, retrained | 0.291 | 0.229 | 0.062 |
| Shipped transformer *(contaminated)* | 0.903 | 0.899 | 0.003 |

Read the last row carefully. A model trained honestly loses a lot of performance when the test set stops sharing scaffolds with the training set — that is what the fingerprint baselines show. The shipped transformer loses almost nothing. That is not evidence that it generalises better than a random forest; it is evidence that the scaffold split did not hold anything back from it, because it had already seen these molecules.

## Fragment prediction

| Model | Valid SMILES | Exact match | Tanimoto | Is a substructure | Rule of 3 |
| --- | --- | --- | --- | --- | --- |
| Nearest neighbour (Tanimoto) | 1.000 | 0.182 | 0.330 | 0.182 | 0.682 |
| Most common fragment | 1.000 | 0.500 | 0.509 | 0.955 | 1.000 |
| **Transformer, retrained on this split** | 0.000 | 0.000 | 0.069 | 0.000 | 0.364 |
| Shipped transformer *(contaminated)* | 0.409 | 0.000 | 0.194 | 0.045 | 0.773 |
| Same recipe, 20,000 training pairs *(intervention 5)* | 0.000 | 0.000 | 0.076 | 0.000 | 0.818 |
| Same recipe trained **on the test set** *(control)* | 0.182 | 0.182 | 0.296 | 0.182 | 0.682 |

**"Is a substructure" is the line that matters.** It is the fraction of returned fragments that are genuinely contained in the molecule they were derived from. A fragment that is not part of its own input is a hallucination, and the pipeline can produce one whenever the model's raw output fails to parse and the nearest-reference fallback substitutes a fragment from some other molecule. Nothing in the deployed code measured this before.

BRICS attachment points (`[5*]`, `[14*]`) are stripped from the reference fragments before any comparison, because the model never emits them; comparing raw strings would score zero by construction rather than by performance.

The control row exists to answer the obvious objection — *was it simply trained badly?* That model was given the test molecules in its training data and trained for 20 epochs. It reproduces 0.182 of the fragments it was shown. So the recipe does learn; what the scaffold split removes is the memorisation it was relying on.

### Intervention 5: more data

The spec's last intervention is more data. Rather than reach for ChEMBL, the original corpus was rebuilt with the pipeline that produced it: `ml-project/FBDD-discovery.py` ran RDKit's BRICS decomposition over a set of drug SMILES, and this repository already holds 8,336 of them in the classification dataset. Running the same decomposition over those yields 0 molecules and 38,433 training pairs, against 1,223 before — the same procedure, no new dependency, nothing to license.

Training was capped at 20,000 pairs for compute, still roughly sixteen times the original corpus.

15 of the new molecules share a scaffold with the held-out validation or test sets and were dropped rather than trained on. The test set is the same 22 molecules as every other fragment row above, so this is a clean comparison: only the training corpus changed.


The fragment test set is **22 molecules**, because the entire supervised fragment corpus in this repository is 219 molecules and 1,223 fragment pairs. Treat these rates as direction, not precision: a single molecule moves any of them by 4.5 percentage points. Growing that corpus is the single highest-value thing that could be done for this model, and it matters more than any architectural change.

## Diagnosing the overfitting

### Learning curve — classification

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 2.1181 | 1.4723 |
| 2 | 1.3032 | 1.4242 |
| 3 | 1.0992 | 1.3536 |
| 4 | 0.8939 | 1.3731 |
| 5 | 0.7451 | 1.5 |
| 6 | 0.6092 | 1.6074 |

Validation loss bottoms out at epoch 3 and rises after it. Everything past epoch 3 is the model memorising the training set. Training was stopped there for every result reported above.

### Learning curve — fragment

| Epoch | Train loss | Validation loss |
| --- | --- | --- |
| 1 | 3.8445 | 2.448 |
| 2 | 2.2881 | 2.1488 |
| 3 | 2.1111 | 2.0968 |
| 4 | 2.0018 | 2.0891 |
| 5 | 1.8996 | 2.0994 |
| 6 | 1.8041 | 2.1158 |

Validation loss bottoms out at epoch 4 and rises after it. Everything past epoch 4 is the model memorising the training set. Training was stopped there for every result reported above.

### Data-scaling curve — classification

| Training data | Compounds | macro f1 |
| --- | --- | --- |
| 10% | 665 | 0.106 |
| 25% | 1663 | 0.097 |
| 50% | 3326 | 0.170 |
| 100% | 6652 | 0.229 |

Still improving at 100% of the data — the bottleneck is data volume, not architecture or regularisation.

### Data-scaling curve — fragment

| Training data | Compounds | mean tanimoto to reference |
| --- | --- | --- |
| 10% | 18 | 0.119 |
| 25% | 44 | 0.106 |
| 50% | 88 | 0.109 |
| 100% | 175 | 0.069 |

Performance has flattened before 100% of the data, so simply adding more of the same data is not the limiting factor here.

### Interventions — classification

Applied one at a time, each measured on its own, in the order the task spec sets out. A combined change would only tell you that *something* helped.

| Intervention | Best epoch | macro f1 | Trainable parameters |
| --- | --- | --- | --- |
| 1 early stopping | 3 | 0.229 | 44,106,239 |
| 2 dropout weight decay | 4 | 0.221 | 44,106,239 |
| 3 class weighting | 3 | 0.229 | 44,106,239 |
| 4 freeze lower layers | 6 | 0.214 | 14,768,639 |

### Interventions — fragment

Applied one at a time, each measured on its own, in the order the task spec sets out. A combined change would only tell you that *something* helped.

| Intervention | Best epoch | substructure rate | Trainable parameters |
| --- | --- | --- | --- |
| 1 early stopping | 4 | 0.000 | 44,106,239 |
| 2 dropout weight decay | 5 | 0.091 | 44,106,239 |
| 4 freeze lower layers | 5 | 0.000 | 14,768,639 |

## Where these models fail, by name

Aggregates let a reader nod along without ever seeing a wrong answer.

**ciprofloxacin** — documented failure: a fluoroquinolone antibiotic the classifier has called antineoplastic  
Dataset label: `antiinfective antineoplastic`  
Raw model output: `antiinfective antineoplastic`  
Prediction after snapping: `antiinfective antineoplastic`  
Fragment model: `CC(=O)1cn(C)c2C)(F)cc1=` (does not parse, **not part of the molecule**)

**aspirin** — small, common, unambiguous anti-inflammatory  
Dataset label: `antiinflammatory cardio cns hematologic`  
Raw model output: `aetaamataliatf` — not a valid label; edit-distance snapping picked the nearest one  
Prediction after snapping: `hematologic`  
Fragment model: `CC(=O)c1ccccc1C` (valid SMILES, **not part of the molecule**)

**caffeine** — CNS stimulant with no therapeutic class in the usual sense  
Dataset label: `cns`  
Raw model output: `cns`  
Prediction after snapping: `cns`  
Fragment model: `Cn1c(=O)c2c(ncn2C)n(C)1=O` (does not parse, **not part of the molecule**)

**imatinib** — kinase inhibitor, antineoplastic  
Dataset label: `antineoplastic`  
Raw model output: `antineoplastic`  
Prediction after snapping: `antineoplastic`  
Fragment model: `CCc1ccc(C(C)=O)1` (valid SMILES, **not part of the molecule**)

**atorvastatin** — lipid-regulating, one of the smaller classes  
Dataset label: `lipidregulating`  
Raw model output: `lipidregulating`  
Prediction after snapping: `lipidregulating`  
Fragment model: `CC(=O)1` (does not parse, **not part of the molecule**)

**tamsulosin** — urological, the rarest class (48 compounds in total)  
Raw model output: `urological`  
Prediction after snapping: `urological`  
Fragment model: `CCOc1ccccc1C` (valid SMILES, **not part of the molecule**)

**salbutamol** — respiratory, a class the models recall poorly  
Dataset label: `reproductivecontrol respiratorysystem`  
Raw model output: `reproductivecontrol resp` — not a valid label; edit-distance snapping picked the nearest one  
Prediction after snapping: `reproductivecontrol`  
Fragment model: `CCC(O)c1ccc(O)c(CO)c1` (valid SMILES, **not part of the molecule**)

## Limitations of this benchmark

- The fragment test set is 22 molecules. Every rate in that section moves by 4.5 percentage points per molecule, so they indicate direction, not precision.
- The retrained models use a fixed, modest training budget. They establish what this recipe achieves under an honest split with this data; they are not a claim about the best achievable number for this architecture.
- The retrained models reproduce the deployed recipe — a masked-LM head decoded position by position and snapped to the nearest label — rather than a sequence-classification head, which would very likely score better. The point was to measure what is in production, and a different architecture would not be comparable with it.
- The remote SMILES corrector the deployed pipeline calls before falling back to nearest-reference snapping is not called here, because a third-party service would make these numbers unreproducible. The intervention rate reported is the rate at which *some* repair was needed.
- The therapeutic-class labels themselves are not audited. Several compounds carry label combinations that are defensible but debatable, and a model is held to the labels it was given. Ciprofloxacin is the documented example: the dataset calls it `antiinfective antineoplastic`, so the model reproducing that is obeying its labels rather than misreading the chemistry.
- The class-weighting intervention scales each batch loss by that batch's mean weight rather than reweighting individual samples. That is a weak form of class weighting, so "it changed nothing" is partly a statement about this implementation and not only about the method.

## Reproducing this

```bash
python3 -m evaluation.splits                       # rebuild and verify the splits
python3 -m evaluation.baselines                    # fingerprint baselines
python3 -m evaluation.evaluate --label shipped     # the published checkpoints
python3 -m evaluation.experiments all --task classification
python3 -m evaluation.experiments all --task fragment
python3 -m evaluation.experiments control --task fragment
python3 -m evaluation.named_examples
python3 -m evaluation.report                       # regenerate this file
python3 tests/test_evaluation.py                   # split integrity and metric tests
```

| Item | Value |
| --- | --- |
| Generated | 2026-11-26 |
| Commit | `b1f718f79a92` |
| classification dataset | `drug_classification_data_df.csv` · sha256 `cb11fab86d265da7` |
| fragment dataset | `ml-project/drug_fragments.csv` · sha256 `692884ab13d7875a` |
| split `classification_scaffold` | `evaluation/splits/classification_scaffold.csv` · sha256 `8896add2516502e2` |
| split `classification_random` | `evaluation/splits/classification_random.csv` · sha256 `8217ef9e9260f8a5` |
| split `fragment_scaffold` | `evaluation/splits/fragment_scaffold.csv` · sha256 `4e64d73e623317aa` |
| split `fragment_random` | `evaluation/splits/fragment_random.csv` · sha256 `bd61af72aea4d15e` |

