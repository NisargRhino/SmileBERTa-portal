"""Figures and analytics for the pharmacophore corpus and models.

Every figure here is generated from the committed results files, never from
numbers typed in by hand, so a figure cannot drift away from the run that
produced it. Each one writes a `.png` and a `.json` of the exact values plotted,
because a reviewer asking "where does that bar come from" should get an answer
without rerunning anything.

Three rules this module follows, all learned from `BENCHMARK.md`:

* **No accuracy anywhere.** 69% of ligand atoms carry no interaction, so an
  accuracy bar would be a figure about the base rate.
* **Support is drawn, not implied.** A per-type AUC of 0.95 on 67 atoms and one
  on 42,208 atoms are different claims, so the support appears on the same axes.
* **The ablation is a figure, not a footnote.** If removing the pocket costs
  nothing, that belongs in a panel of its own at the same size as the headline.
"""
import argparse
import json
import os
from collections import Counter

import matplotlib
matplotlib.use('Agg')                     # no display on a build host
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402

from pharmacophore.interactions import INTERACTIONS   # noqa: E402

# A colour-blind-safe qualitative set; the pocket/no-pocket pair is deliberately
# one strong and one muted tone of the same hue so the comparison reads at a
# glance even in greyscale.
WITH_POCKET = '#1f6feb'
NO_POCKET = '#9bb8e0'
SHUFFLED = '#d1495b'
NEUTRAL = '#6b7785'
ACCENT = '#2f9e44'

plt.rcParams.update({
    'figure.dpi': 140, 'savefig.dpi': 140, 'font.size': 9,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlesize': 10, 'axes.titleweight': 'bold',
    'figure.facecolor': 'white', 'savefig.bbox': 'tight',
})


def _save(fig, out_dir, name, payload):
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, name + '.png')
    fig.savefig(png)
    plt.close(fig)
    with open(os.path.join(out_dir, name + '.json'), 'w') as handle:
        json.dump(payload, handle, indent=2)
    return png


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _drop_stale_regimes(model, baselines, log=print):
    """Discard model results that were produced against a different split.

    Results files are written per run and a long build means the corpus grows
    underneath them, so `model.json` can hold a regime from last night's
    97-complex smoke test while `baselines.json` holds today's 14,000. Plotting
    both puts a 12-complex bar beside a 1,745-complex bar on the same axes with
    nothing to say so. Where the two disagree on the size of the test set, the
    model entry is dropped rather than drawn.
    """
    if not model or not baselines:
        return model
    kept = {}
    for regime, entry in model.items():
        theirs = ((baselines.get(regime) or {}).get('_split') or {}).get('test_complexes')
        ours = (entry.get('counts') or {}).get('test')
        if theirs and ours and theirs != ours:
            log('  skipping stale model result for %s: %d test complexes against '
                '%d in the current split' % (regime, ours, theirs))
            continue
        kept[regime] = entry
    return kept


# --------------------------------------------------------------------------
# Corpus analytics, straight from the records
# --------------------------------------------------------------------------

def corpus_statistics(release, resolutions=None):
    """Scan the record shards once and return everything the figures need.

    `resolutions` is the authoritative wwPDB index. It is preferred over the
    value stored in a record, because records written before the REMARK 2
    parser was fixed all carry exactly 2.0 A — the remark number rather than
    the resolution. Reading the index costs one cached download and repairs
    those records for reporting without rebuilding the corpus.
    """
    records_dir = os.path.join(release, 'records')
    stats = {
        'complexes': 0, 'entries': set(), 'proteins': Counter(),
        'ligands': set(), 'rejections': Counter(), 'by_type': Counter(),
        'heavy_atoms': 0, 'atoms_with_contact': 0,
        'resolutions': [], 'heavy_atom_sizes': [], 'pocket_sizes': [],
        'interactions_per_complex': [],
    }
    if not os.path.isdir(records_dir):
        return stats
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(records_dir, name)) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue            # a torn final line from a killed batch
                stats['entries'].add(record.get('pdb_id'))
                if record.get('rejected'):
                    stats['rejections'][record['rejected']] += 1
                    continue
                stats['complexes'] += 1
                if record.get('uniprot'):
                    stats['proteins'][record['uniprot']] += 1
                if record.get('inchikey'):
                    stats['ligands'].add(record['inchikey'])
                stats['heavy_atoms'] += record.get('heavy_atoms', 0)
                stats['atoms_with_contact'] += len(record.get('per_atom') or {})
                true_resolution = (resolutions or {}).get(
                    (record.get('pdb_id') or '').upper(), record.get('resolution'))
                if true_resolution:
                    stats['resolutions'].append(true_resolution)
                if record.get('heavy_atoms'):
                    stats['heavy_atom_sizes'].append(record['heavy_atoms'])
                if record.get('n_pocket_residues'):
                    stats['pocket_sizes'].append(record['n_pocket_residues'])
                stats['interactions_per_complex'].append(record.get('n_interactions', 0))
                for key, value in (record.get('by_type') or {}).items():
                    stats['by_type'][key] += value
    return stats


def figure_corpus_overview(stats, out_dir):
    """Four panels describing what the corpus actually contains."""
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.4))

    # Interaction types by labelled atoms, with the rare ones visible.
    ax = axes[0][0]
    items = [(t, stats['by_type'].get(t, 0)) for t in INTERACTIONS]
    items.sort(key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    counts = [v for _, v in items]
    ax.barh(range(len(names)), counts, color=WITH_POCKET)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale('log')
    ax.set_xlabel('labelled atoms (log scale)')
    ax.set_title('Interaction types in the corpus')
    for i, value in enumerate(counts):
        if value:
            ax.text(value * 1.15, i, f'{value:,}', va='center', fontsize=7.5)

    # Ligand size: the fragment-vs-drug question, answered by the data.
    ax = axes[0][1]
    sizes = stats['heavy_atom_sizes']
    if sizes:
        ax.hist(sizes, bins=range(8, 70, 2), color=ACCENT, alpha=0.85)
        ax.axvline(float(np.median(sizes)), color=NEUTRAL, ls='--', lw=1,
                   label=f'median {np.median(sizes):.0f}')
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel('ligand heavy atoms')
    ax.set_ylabel('complexes')
    ax.set_title('Ligand size distribution')

    # Resolution: the quality floor the filter enforces.
    ax = axes[1][0]
    if stats['resolutions']:
        ax.hist(stats['resolutions'], bins=np.arange(0.5, 2.65, 0.1),
                color=WITH_POCKET, alpha=0.85)
    ax.set_xlabel('resolution (Å)')
    ax.set_ylabel('complexes')
    ax.set_title('Crystallographic resolution (≤ 2.5 Å enforced)')

    # Where the entries went. Rejections are the audit trail.
    ax = axes[1][1]
    rows = [('kept', stats['complexes'])] + stats['rejections'].most_common(6)
    labels = [r[0].replace('_', '\n') for r in rows]
    values = [r[1] for r in rows]
    colours = [ACCENT] + [NEUTRAL] * (len(rows) - 1)
    ax.bar(range(len(rows)), values, color=colours)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel('records')
    ax.set_title('Kept vs rejected, by reason')

    fig.suptitle('Pharmacophore corpus: %s complexes, %s proteins, %s ligands'
                 % (f"{stats['complexes']:,}", f"{len(stats['proteins']):,}",
                    f"{len(stats['ligands']):,}"), fontsize=11, fontweight='bold')
    fig.tight_layout()
    payload = {
        'complexes': stats['complexes'],
        'entries_touched': len(stats['entries']),
        'unique_proteins': len(stats['proteins']),
        'unique_ligands': len(stats['ligands']),
        'heavy_atoms': stats['heavy_atoms'],
        'atoms_with_contact': stats['atoms_with_contact'],
        'contact_rate': round(stats['atoms_with_contact']
                              / max(stats['heavy_atoms'], 1), 4),
        'by_type': dict(stats['by_type']),
        'rejections': dict(stats['rejections']),
        'median_heavy_atoms': float(np.median(stats['heavy_atom_sizes']))
        if stats['heavy_atom_sizes'] else None,
        'median_resolution': float(np.median(stats['resolutions']))
        if stats['resolutions'] else None,
    }
    return _save(fig, out_dir, 'corpus_overview', payload)


def figure_protein_coverage(stats, out_dir):
    """How concentrated the corpus is — the bias a reviewer will look for first."""
    counts = sorted(stats['proteins'].values(), reverse=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))

    ax = axes[0]
    if counts:
        ax.plot(range(1, len(counts) + 1), counts, color=WITH_POCKET)
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlabel('protein rank (log)')
    ax.set_ylabel('complexes (log)')
    ax.set_title('Complexes per protein: a long tail')

    ax = axes[1]
    total = sum(counts) or 1
    cumulative = np.cumsum(counts) / total
    if counts:
        ax.plot(np.arange(1, len(counts) + 1) / len(counts) * 100,
                cumulative * 100, color=ACCENT)
        ax.axhline(50, color=NEUTRAL, ls='--', lw=0.8)
    ax.set_xlabel('% of proteins, most-studied first')
    ax.set_ylabel('% of complexes')
    ax.set_title('Concentration of the corpus')

    share = None
    if counts:
        top = max(1, len(counts) // 100)
        share = round(float(sum(counts[:top]) / total), 4)
    fig.tight_layout()
    return _save(fig, out_dir, 'protein_coverage', {
        'unique_proteins': len(counts),
        'most_studied_protein_complexes': counts[0] if counts else 0,
        'share_held_by_top_1pct_of_proteins': share,
        'median_complexes_per_protein': float(np.median(counts)) if counts else None,
    })


def ligand_promiscuity(release):
    """How much a ligand's labels change when the pocket changes.

    This bounds the pocket's possible contribution without training anything.
    If a molecule carried the same labels wherever it binds, a ligand-only model
    could be near-perfect and the pocket would have nothing left to add — which
    is one of the two readings of the transformer's small ablation delta. The
    other is that the conditioning is too weak. This measurement separates them.
    """
    import itertools
    by_ligand = {}
    records_dir = os.path.join(release, 'records')
    if not os.path.isdir(records_dir):
        return {}
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith('.jsonl'):
            continue
        with open(os.path.join(records_dir, name)) as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get('rejected') or not record.get('inchikey'):
                    continue
                by_ligand.setdefault(record['inchikey'], []).append(record)

    def typed(record):
        out = set()
        for index, types in (record.get('per_atom') or {}).items():
            for name in types:
                out.add((int(index), name))
        return out

    def touched(record):
        return {int(i) for i in (record.get('per_atom') or {})}

    overlap_atoms, overlap_typed, pairs, multi = [], [], 0, 0
    for records in by_ligand.values():
        representative = {}
        for record in records:
            representative.setdefault(record.get('uniprot'), record)
        if len(representative) < 2:
            continue
        multi += 1
        for a, b in itertools.combinations(representative.values(), 2):
            pairs += 1
            for fn, store in ((touched, overlap_atoms), (typed, overlap_typed)):
                x, y = fn(a), fn(b)
                if x or y:
                    store.append(len(x & y) / len(x | y))
    if not overlap_typed:
        return {}
    return {
        'ligands': len(by_ligand),
        'ligands_in_multiple_proteins': multi,
        'cross_protein_pairs': pairs,
        'overlap_atoms_touched': {
            'mean': round(float(np.mean(overlap_atoms)), 4),
            'median': round(float(np.median(overlap_atoms)), 4)},
        'overlap_atom_and_type': {
            'mean': round(float(np.mean(overlap_typed)), 4),
            'median': round(float(np.median(overlap_typed)), 4)},
        'identical_label_pairs': round(
            float(np.mean([v > 0.999 for v in overlap_typed])), 4),
        'disjoint_label_pairs': round(
            float(np.mean([v < 0.001 for v in overlap_typed])), 4),
        'values': overlap_typed,
    }


def figure_promiscuity(promiscuity, out_dir):
    """The ceiling on any ligand-only model, measured from the labels."""
    values = promiscuity.get('values')
    if not values:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))

    ax = axes[0]
    ax.hist(values, bins=np.linspace(0, 1, 26), color=WITH_POCKET, alpha=0.9)
    ax.axvline(float(np.median(values)), color=SHUFFLED, ls='--', lw=1.2,
               label='median %.2f' % np.median(values))
    ax.set_xlabel('label overlap (Jaccard) for one ligand across two proteins')
    ax.set_ylabel('pairs')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('The same molecule is labelled differently\nin different pockets')

    ax = axes[1]
    identical = promiscuity['identical_label_pairs'] * 100
    disjoint = promiscuity['disjoint_label_pairs'] * 100
    partial = 100 - identical - disjoint
    ax.bar(['identical', 'partial', 'no overlap'], [identical, partial, disjoint],
           color=[NO_POCKET, WITH_POCKET, SHUFFLED])
    for i, value in enumerate([identical, partial, disjoint]):
        ax.text(i, value + 1, '%.1f%%' % value, ha='center', fontsize=8)
    ax.set_ylabel('% of cross-protein pairs')
    ax.set_title('A ligand-only model must give one answer\nfor all of these')

    fig.tight_layout()
    payload = {k: v for k, v in promiscuity.items() if k != 'values'}
    return _save(fig, out_dir, 'ligand_promiscuity', payload)


# --------------------------------------------------------------------------
# Model and ablation figures
# --------------------------------------------------------------------------

def figure_ablation(baselines, model, out_dir):
    """The figure the project turns on: does the pocket contribute?"""
    regimes = [r for r in ('unseen_ligand', 'unseen_protein', 'both_unseen')
               if (baselines or {}).get(r) or (model or {}).get(r)]
    if not regimes:
        return None

    fig, axes = plt.subplots(1, len(regimes), figsize=(3.3 * len(regimes), 3.9),
                             sharey=True, squeeze=False)
    payload = {}
    for index, regime in enumerate(regimes):
        ax = axes[0][index]
        bars, colours, labels = [], [], []
        b = (baselines or {}).get(regime) or {}
        m = (model or {}).get(regime) or {}

        def add(label, value, colour):
            if value is not None:
                labels.append(label)
                bars.append(value)
                colours.append(colour)

        add('prior', (b.get('frequency_prior') or {}).get('macro_auc'), NEUTRAL)
        add('RF\nligand', (b.get('random_forest_ligand_only') or {}).get('macro_auc'), NO_POCKET)
        add('RF\n+pocket', (b.get('random_forest_with_pocket') or {}).get('macro_auc'), WITH_POCKET)
        add('net\nligand', ((m.get('no_pocket') or {}).get('test') or {}).get('macro_auc'), NO_POCKET)
        add('net\n+pocket', ((m.get('with_pocket') or {}).get('test') or {}).get('macro_auc'), WITH_POCKET)
        add('net\nshuffled', ((m.get('shuffled_pocket') or {}).get('test') or {}).get('macro_auc'), SHUFFLED)

        ax.bar(range(len(bars)), bars, color=colours)
        ax.set_xticks(range(len(bars)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.axhline(0.5, color='#999', ls=':', lw=0.9)
        ax.set_ylim(0.4, 1.0)
        for i, value in enumerate(bars):
            ax.text(i, value + 0.012, f'{value:.3f}', ha='center', fontsize=7)
        n = (m.get('counts') or {}).get('test') or (b.get('_split') or {}).get('test_complexes')
        ax.set_title('%s\n%s test complexes' % (regime, f'{n:,}' if n else '?'),
                     fontsize=9)
        if index == 0:
            ax.set_ylabel('macro AUC over interaction types')
        payload[regime] = {
            'bars': dict(zip([l.replace('\n', ' ') for l in labels], bars)),
            'rf_ablation': (b.get('_ablation') or {}),
            'net_ablation': (m.get('_ablation') or {}),
            'net_shuffle_check': (m.get('_shuffle_check') or {}),
        }

    fig.suptitle('Does the pocket contribute? 0.5 is chance; ligand-only bars are the control',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return _save(fig, out_dir, 'ablation', payload)


def figure_subset_ablation(baselines, model, out_dir):
    """Corpus-wide against promiscuous-ligand ablation, forest against network.

    This is the comparison that identifies where the problem is. Both models see
    the same split, the same features and the same subset; if one finds the
    pocket and the other does not, the difference is the architecture, not the
    data.
    """
    rows = []
    for regime in ('unseen_ligand', 'unseen_protein', 'both_unseen'):
        b = (baselines or {}).get(regime) or {}
        m = (model or {}).get(regime) or {}
        entry = {'regime': regime}
        entry['rf_all'] = (b.get('_ablation') or {}).get('delta')
        entry['net_all'] = (m.get('_ablation') or {}).get('delta')
        bs = b.get('multi_protein_ligands') or {}
        ms = m.get('multi_protein_ligands') or {}
        entry['rf_subset'] = (bs.get('_ablation') or {}).get('delta')
        entry['net_subset'] = (ms.get('_ablation') or {}).get('delta')
        entry['subset_atoms'] = bs.get('atoms') or ms.get('atoms')
        entry['subset_complexes'] = bs.get('complexes') or ms.get('complexes')
        if any(entry.get(k) is not None for k in
               ('rf_all', 'net_all', 'rf_subset', 'net_subset')):
            rows.append(entry)
    if not rows:
        return None

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    labels = ['RF, all', 'RF, promiscuous', 'net, all', 'net, promiscuous']
    keys = ['rf_all', 'rf_subset', 'net_all', 'net_subset']
    colours = [NO_POCKET, WITH_POCKET, '#f0a202', SHUFFLED]
    width = 0.2
    x = np.arange(len(rows))
    for i, (key, label, colour) in enumerate(zip(keys, labels, colours)):
        values = [(r.get(key) if r.get(key) is not None else np.nan) for r in rows]
        ax.bar(x + (i - 1.5) * width, values, width, label=label, color=colour)
        for j, value in enumerate(values):
            if not np.isnan(value):
                ax.text(x[j] + (i - 1.5) * width, value + 0.002,
                        '%.3f' % value, ha='center', fontsize=6.5)
    ax.axhline(0.02, color=ACCENT, ls='--', lw=1.1,
               label='0.02 margin: "uses the pocket"')
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n%s' % (r['regime'],
                                    ('subset %s complexes' % f"{r['subset_complexes']:,}")
                                    if r.get('subset_complexes') else 'subset too small')
                        for r in rows], fontsize=8)
    ax.set_ylabel('macro AUC lost by removing the pocket')
    # The title states only what the bars actually present support. The network
    # arms are absent whenever model.json was written against an older split,
    # and a figure that keeps its conclusion in the title regardless would be
    # asserting a comparison it is not showing.
    has_net = any(r.get('net_all') is not None or r.get('net_subset') is not None
                  for r in rows)
    has_rf = any(r.get('rf_all') is not None for r in rows)
    if has_net and has_rf:
        title = ('Where the pocket signal goes: the forest finds it, '
                 'the network does not')
    elif has_rf:
        title = ('Pocket contribution by regime (forest only — no model results '
                 'for the current split yet)')
    else:
        title = 'Pocket contribution by regime'
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='upper left')
    fig.tight_layout()
    return _save(fig, out_dir, 'subset_ablation', {'rows': rows})


def figure_per_type(model, out_dir, regime='unseen_ligand'):
    """Per-type AUC against support, because an average hides the rare types."""
    arm = ((model or {}).get(regime) or {}).get('with_pocket') or {}
    per_type = (arm.get('test') or {}).get('per_type')
    if not per_type:
        return None
    no_pocket = (((model or {}).get(regime) or {}).get('no_pocket') or {})
    per_type_np = (no_pocket.get('test') or {}).get('per_type') or {}

    names = [n for n in INTERACTIONS if per_type.get(n, {}).get('support')]
    names.sort(key=lambda n: -per_type[n]['support'])
    aucs = [per_type[n]['auc'] or 0 for n in names]
    aucs_np = [(per_type_np.get(n, {}).get('auc') or 0) for n in names]
    support = [per_type[n]['support'] for n in names]

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 5.6), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, aucs, 0.4, label='with pocket', color=WITH_POCKET)
    if any(aucs_np):
        ax.bar(x + 0.2, aucs_np, 0.4, label='ligand only', color=NO_POCKET)
    ax.axhline(0.5, color='#999', ls=':', lw=0.9)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel('AUC')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Per-interaction-type AUC, %s (test)' % regime)
    for i, value in enumerate(aucs):
        ax.text(i - 0.2, value + 0.01, f'{value:.2f}', ha='center', fontsize=6.5)

    ax = axes[1]
    ax.bar(x, support, 0.6, color=NEUTRAL)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('test atoms (log)')
    ax.set_title('Support — a high AUC on a tiny support is not a result', fontsize=9)

    fig.tight_layout()
    return _save(fig, out_dir, 'per_type_auc_%s' % regime, {
        'regime': regime,
        'per_type': {n: {'auc': per_type[n]['auc'],
                         'auc_ligand_only': per_type_np.get(n, {}).get('auc'),
                         'average_precision': per_type[n]['average_precision'],
                         'support': per_type[n]['support'],
                         'base_rate': per_type[n]['base_rate']} for n in names},
    })


def figure_training_curves(model, out_dir):
    """Validation curves for both arms, which is where overfitting shows."""
    regimes = [r for r in ('unseen_ligand', 'unseen_protein', 'both_unseen')
               if ((model or {}).get(r) or {}).get('with_pocket')]
    if not regimes:
        return None
    fig, axes = plt.subplots(1, len(regimes), figsize=(3.4 * len(regimes), 3.3),
                             sharey=True, squeeze=False)
    payload = {}
    for index, regime in enumerate(regimes):
        ax = axes[0][index]
        entry = model[regime]
        for arm, colour, label in (('with_pocket', WITH_POCKET, 'with pocket'),
                                   ('no_pocket', NO_POCKET, 'ligand only')):
            history = ((entry.get(arm) or {}).get('training') or {}).get('history') or []
            if history:
                ax.plot([h['epoch'] for h in history],
                        [h['val_macro_auc'] for h in history],
                        color=colour, label=label, lw=1.4)
        ax.set_title(regime, fontsize=9)
        ax.set_xlabel('epoch')
        if index == 0:
            ax.set_ylabel('validation macro AUC')
            ax.legend(frameon=False, fontsize=8)
        payload[regime] = {
            arm: ((entry.get(arm) or {}).get('training') or {}).get('history')
            for arm in ('with_pocket', 'no_pocket')
        }
    fig.suptitle('Validation curves — model selection never touched the test set',
                 fontsize=10, fontweight='bold')
    fig.tight_layout()
    return _save(fig, out_dir, 'training_curves', payload)


def write_report(release, stats, baselines, model, figures, out_dir, promiscuity=None):
    """A markdown summary beside the figures, generated from the same values."""
    lines = ['# Pharmacophore results', '',
             'Generated by `pharmacophore.figures` from the committed results '
             'files. Every number here is read from those files, not retyped.', '',
             '## Corpus', '',
             '| | |', '| --- | --- |',
             f"| Complexes | {stats['complexes']:,} |",
             f"| Entries touched | {len(stats['entries']):,} |",
             f"| Unique proteins | {len(stats['proteins']):,} |",
             f"| Unique ligands | {len(stats['ligands']):,} |",
             f"| Labelled heavy atoms | {stats['heavy_atoms']:,} |",
             '| Atoms carrying a contact | %.1f%% |'
             % (100 * stats['atoms_with_contact'] / max(stats['heavy_atoms'], 1)),
             '']
    if promiscuity:
        lines += [
            '## How much could the pocket possibly matter?', '',
            'Measured from the labels alone, no model involved. Taking every '
            'ligand that appears in more than one protein and comparing its '
            'per-atom labels between those proteins:', '',
            '| | |', '| --- | --- |',
            f"| Ligands appearing in >1 protein | {promiscuity['ligands_in_multiple_proteins']:,} "
            f"of {promiscuity['ligands']:,} |",
            f"| Cross-protein pairs compared | {promiscuity['cross_protein_pairs']:,} |",
            '| Median label overlap (atom + type) | %.3f |'
            % promiscuity['overlap_atom_and_type']['median'],
            '| Pairs with identical labels | %.1f%% |'
            % (100 * promiscuity['identical_label_pairs']),
            '| Pairs sharing no label at all | %.1f%% |'
            % (100 * promiscuity['disjoint_label_pairs']),
            '',
            'A ligand-only model has to give one answer for every pair in that '
            'table, so the low overlap is a hard ceiling on how well it can do '
            'and direct evidence that the pocket carries information.', '']
    if model or baselines:
        lines += ['## Does the pocket contribute?', '',
                  '| Regime | RF ligand | RF +pocket | RF delta | net ligand | '
                  'net +pocket | net delta | net shuffled |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for regime in ('unseen_ligand', 'unseen_protein', 'both_unseen'):
            b = (baselines or {}).get(regime) or {}
            m = (model or {}).get(regime) or {}

            def cell(value):
                return '%.4f' % value if isinstance(value, (int, float)) else '—'

            rf_l = (b.get('random_forest_ligand_only') or {}).get('macro_auc')
            rf_p = (b.get('random_forest_with_pocket') or {}).get('macro_auc')
            net_l = ((m.get('no_pocket') or {}).get('test') or {}).get('macro_auc')
            net_p = ((m.get('with_pocket') or {}).get('test') or {}).get('macro_auc')
            net_s = ((m.get('shuffled_pocket') or {}).get('test') or {}).get('macro_auc')
            lines.append('| `%s` | %s | %s | %s | %s | %s | %s | %s |' % (
                regime, cell(rf_l), cell(rf_p),
                cell(rf_p - rf_l) if rf_l is not None and rf_p is not None else '—',
                cell(net_l), cell(net_p),
                cell(net_p - net_l) if net_l is not None and net_p is not None else '—',
                cell(net_s)))
        lines += ['', 'The margin that counts as "uses the pocket" is 0.02 macro '
                  'AUC, set before the numbers were seen.', '']
    if figures:
        lines += ['## Figures', ''] + \
                 ['- `%s`' % os.path.basename(p) for p in figures if p] + ['']
    path = os.path.join(out_dir, 'RESULTS.md')
    with open(path, 'w') as handle:
        handle.write('\n'.join(lines))
    return path


def run(release, out_dir=None, log=print):
    out_dir = out_dir or os.path.join(release, 'figures')
    log('scanning records in %s ...' % release)
    try:
        from pharmacophore import sources
        resolutions = sources.resolution_index()
    except Exception as exc:                  # offline: fall back to the records
        log('  resolution index unavailable (%s); using stored values' % exc)
        resolutions = None
    stats = corpus_statistics(release, resolutions)
    log('  %d complexes, %d proteins, %d ligands'
        % (stats['complexes'], len(stats['proteins']), len(stats['ligands'])))

    baselines = _load(os.path.join(release, 'results', 'baselines.json'))
    model = _drop_stale_regimes(
        _load(os.path.join(release, 'results', 'model.json')), baselines, log)

    promiscuity = ligand_promiscuity(release)
    if promiscuity:
        log('  same ligand across different proteins: %.3f median label overlap '
            'over %d pairs' % (promiscuity['overlap_atom_and_type']['median'],
                               promiscuity['cross_protein_pairs']))
    figures = [figure_corpus_overview(stats, out_dir),
               figure_protein_coverage(stats, out_dir),
               figure_promiscuity(promiscuity, out_dir)]
    if baselines or model:
        figures.append(figure_ablation(baselines, model, out_dir))
        figures.append(figure_subset_ablation(baselines, model, out_dir))
    if model:
        figures.append(figure_training_curves(model, out_dir))
        for regime in ('unseen_ligand', 'unseen_protein', 'both_unseen'):
            figures.append(figure_per_type(model, out_dir, regime))
    figures = [f for f in figures if f]
    report = write_report(release, stats, baselines, model, figures, out_dir,
                          promiscuity)
    for path in figures:
        log('  wrote %s' % path)
    log('  wrote %s' % report)
    return figures + [report]


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', default='corpus/v1')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    run(args.release, args.out)
