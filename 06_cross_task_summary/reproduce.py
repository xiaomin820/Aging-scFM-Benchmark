"""Recompute published summary rankings and AP from the bundled evidence.

This release entry point is new; original training and upstream GRN inference
are not run. The historical aggregate-then-rank implementation is reused.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from bootstrap_ap import weighted_bootstrap_ap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '00_shared/evaluation'))
import cross_task_analysis as original


def require(condition, message):
    if not condition:
        raise ValueError(message)


def table(data, number):
    matches = list((data / 'supplementary').glob(f'Supplementary Table S{number}.*.csv'))
    require(len(matches) == 1, f'Expected one S{number} table, found {len(matches)}')
    return pd.read_csv(matches[0])


def compare_frames(actual, expected, key, tolerance, label):
    require(not actual[key].duplicated().any(), f'Duplicate calculated keys: {label}')
    require(not expected[key].duplicated().any(), f'Duplicate published keys: {label}')
    a, e = actual.set_index(key), expected.set_index(key)
    require(set(a.index) == set(e.index), f'Model coverage differs: {label}')
    a = a.reindex(e.index)
    for column in e.columns:
        require(column in a, f'Missing column {label}.{column}')
        av = pd.to_numeric(a[column], errors='raise').to_numpy(dtype=float)
        ev = pd.to_numeric(e[column], errors='raise').to_numpy(dtype=float)
        require(np.allclose(av, ev, atol=tolerance, rtol=0, equal_nan=True),
                f'Published value mismatch: {label}.{column}')


def rankings(data, output, config):
    tables, _, issues = original.load_tables(data / 'supplementary')
    require(not issues, f'Inputs need explicit curation; refusing inferred model IDs: {issues}')
    records = original.build_records(tables)
    for task, count in config['expected_contexts'].items():
        require(records.loc[records.task == task, 'context'].nunique() == count,
                f'Unexpected number of contexts in {task}')
    scores, diagnostic = original.aggregate(records, original.MODEL_ORDER, require_all=False)
    for task, count in config['expected_complete_methods'].items():
        require(scores[task].notna().sum() == count, f'Incomplete method coverage in {task}')
    metrics = dict(T1='mean_donor_PCC', T2='mean_signed_PCC',
                   T3='positive_DeltaAgeGap_fraction', T4='mean_F1', T5='AUPRC')
    summary = pd.DataFrame({'model': [original.DISPLAY_NAMES.get(m, m) for m in scores.index]})
    for task in original.TASKS:
        subset = diagnostic[diagnostic.task == task].drop_duplicates('model').set_index('model')
        for name, field in [(f'{task}_{metrics[task]}', 'task_metric_mean'),
                            (f'{task}_rank', 'task_rank_by_metric_mean'),
                            (f'{task}_normalized_score', 'task_score')]:
            summary[name] = subset[field].reindex(scores.index).to_numpy()
    compare_frames(summary, table(data, 15), 'model', config['published_table_rounding_tolerance'], 'S15')

    fm_scores, fm_details = original.aggregate(records, original.FOUNDATION_MODELS)
    composite = pd.DataFrame({'model': original.FOUNDATION_MODELS})
    for task in original.TASKS:
        details = fm_details[fm_details.task == task].drop_duplicates('model').set_index('model')
        composite[f'{task}_score'] = fm_scores[task].reindex(composite.model).to_numpy()
        composite[f'{task}_rank'] = details.task_rank_by_metric_mean.reindex(composite.model).to_numpy()
    # Average ranks are integers/half-integers: derive the mean score from their
    # mean to preserve exact ties across different task-rank combinations.
    composite['mean_task_rank'] = composite[[f'{t}_rank' for t in original.TASKS]].mean(axis=1)
    composite['mean_score'] = (10 - composite.mean_task_rank) / 9
    composite['overall_rank'] = composite.mean_task_rank.rank(method='average')
    compare_frames(composite, table(data, 16), 'model', config['published_table_rounding_tolerance'], 'S16')
    composite = composite.sort_values(['overall_rank', 'model'])
    records.to_csv(output / 'task_context_metrics.csv', index=False)
    diagnostic.to_csv(output / 'task_ranking_details.csv', index=False)
    summary.to_csv(output / 'S15_recomputed.csv', index=False)
    composite.to_csv(output / 'S16_recomputed.csv', index=False)
    for task in original.TASKS:
        summary[['model', *[c for c in summary if c.startswith(task + '_')]]].dropna(how='all', subset=[f'{task}_rank']).to_csv(output / f'{task}_summary.csv', index=False)
    return scores, len(records)


def task5(data, output, config):
    directory = data / 'task5'
    expected = pd.read_csv(directory / 'pre_motif_AP_all_references.csv')
    require(not expected.duplicated(['model_id', 'reference']).any(), 'Duplicate T5 reference rows')
    require(len(expected) == 33, 'Expected eleven methods and three references')
    genes = pd.read_csv(directory / 'common_genes.csv').gene.tolist()
    tfs = pd.read_csv(directory / 'common_TFs.csv').TF.tolist()
    require(genes == sorted(set(genes)) and tfs == sorted(set(tfs)), 'Candidate names must be sorted and unique')
    require(len(genes) == 1571 and len(tfs) == 104 and set(tfs).issubset(genes), 'Unexpected candidate universe')
    pairs = [(tf, gene) for tf in tfs for gene in genes if tf != gene]
    edge_tf = np.array([i for i, tf in enumerate(tfs) for gene in genes if tf != gene], dtype=np.int32)
    multiplicities = np.load(directory / 'bootstrap_TF_multiplicities.npy', allow_pickle=False)
    require(multiplicities.shape == (config['task5_bootstrap_replicates'], len(tfs)), 'Invalid TF resampling shape')
    require(np.issubdtype(multiplicities.dtype, np.integer) and (multiplicities >= 0).all(), 'Invalid TF multiplicities')
    require((multiplicities.sum(axis=1) == len(tfs)).all(), 'TF resamples must contain 104 draws')
    references = {'Union': 'Union_reference_edges.csv', 'TRRUST': 'TRRUST_reference_edges.csv',
                  'DoRothEA A-C': 'DoRothEA_ABC_reference_edges.csv'}
    methods = set(original.FOUNDATION_MODELS + ['SCENIC'])
    require(set(zip(expected.model_id, expected.reference)) == {(m, r) for m in methods for r in references},
            'T5 requires all eleven methods and all three references')
    rows, regenerated = [], {}
    with threadpool_limits(limits=1), np.load(directory / 'scores_and_labels.npz', allow_pickle=False) as arrays, \
         np.load(directory / 'bootstrap_AP.npz', allow_pickle=False) as draws:
        for reference, name in references.items():
            edges = pd.read_csv(directory / name)
            edge_set = set(zip(edges.TF, edges.Target))
            require(len(edge_set) == len(edges), f'Duplicate reference edges: {reference}')
            reconstructed = np.array([int(pair in edge_set) for pair in pairs], dtype=np.int32)
            require(np.array_equal(reconstructed, arrays[f'labels_{reference}']), f'Reference label alignment mismatch: {reference}')
            regenerated[f'SCENIC__{reference}'] = weighted_bootstrap_ap(
                arrays['SCENIC'], arrays[f'labels_{reference}'], edge_tf, multiplicities)
        for row in expected.to_dict('records'):
            model, reference = row['model_id'], row['reference']
            y, x = arrays[f'labels_{reference}'], arrays[model]
            require(y.shape == x.shape == (config['task5_candidate_pairs'],), 'T5 candidate shape mismatch')
            require(set(np.unique(y)) == {0, 1} and np.isfinite(x).all(), 'Invalid labels or scores')
            require(int(y.sum()) == int(row['positive_edges']), 'Reference positive count mismatch')
            ap = float(average_precision_score(y, x))
            key = f'{model}__{reference}'
            if key not in regenerated:
                regenerated[key] = weighted_bootstrap_ap(x, y, edge_tf, multiplicities)
            boot = regenerated[key]
            base = regenerated[f'SCENIC__{reference}']
            require(boot.shape == base.shape == (config['task5_bootstrap_replicates'],), 'Bootstrap shape mismatch')
            require(np.isfinite(boot).all() and np.isfinite(base).all(), 'Nonfinite bootstrap AP')
            bootstrap_error = float(np.max(np.abs(boot - draws[key])))
            require(bootstrap_error <= config['ap_absolute_tolerance'], f'Bootstrap AP mismatch: {model}/{reference}')
            lower, upper = np.percentile(boot, [2.5, 97.5])
            delta_lower, delta_upper = np.percentile(boot - base, [2.5, 97.5])
            baseline_ap = average_precision_score(y, arrays['SCENIC'])
            calculated = dict(AP=ap, AP_CI_lower=lower, AP_CI_upper=upper,
                              AP_difference_vs_baseline=ap-baseline_ap,
                              difference_CI_lower=delta_lower, difference_CI_upper=delta_upper)
            for key, value in calculated.items():
                require(abs(value - row[key]) <= config['ap_absolute_tolerance'], f'T5 mismatch: {model}/{reference}/{key}')
            rows.append(dict(model_id=model, model=row['model'], reference=reference,
                             **calculated, AUROC_recomputed=roc_auc_score(y, x),
                             candidate_pairs=len(y), positive_edges=int(y.sum()),
                             AP_over_prevalence=ap/float(y.mean()),
                             AP_absolute_error=abs(ap-row['AP']),
                             bootstrap_AP_max_absolute_error=bootstrap_error))
        require(int(arrays['labels_Union'].sum()) == config['task5_union_positives'], 'Unexpected Union positives')
    result = pd.DataFrame(rows)
    union = result[result.reference == 'Union'].set_index('model_id')
    primary = pd.read_csv(directory / 'primary_AP.csv').set_index('model_id')
    s17 = table(data, 17).set_index('model')
    require(set(union.index) == set(primary.index) == set(s17.index), 'T5 method sets differ')
    for model in union.index:
        for value in [primary.loc[model, 'AP'], s17.loc[model, 'AP'], s17.loc[model, 'AUPRC']]:
            require(abs(union.loc[model, 'AP']-value) <= config['ap_absolute_tolerance'], f'S17 AP mismatch: {model}')
        require(abs(union.loc[model, 'AUROC_recomputed']-s17.loc[model, 'AUROC']) <= 5.1e-5,
                f'S17 AUROC mismatch: {model}')
        for column in ['AP_CI_lower', 'AP_CI_upper', 'AP_difference_vs_baseline',
                       'difference_CI_lower', 'difference_CI_upper', 'AP_over_prevalence']:
            require(abs(union.loc[model, column]-s17.loc[model, column]) <= config['ap_absolute_tolerance'],
                    f'S17 mismatch: {model}/{column}')
    np.savez_compressed(output / 'T5_bootstrap_AP_recomputed.npz', **regenerated)
    result.to_csv(output / 'T5_AP_recomputed.csv', index=False)
    return result


def plots(scores, ap, output):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.hashsalt': 'aging-benchmark'})
    fig, ax = plt.subplots(figsize=(10, 5))
    matrix = scores.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap='viridis', aspect='auto')
    ax.set_xticks(range(5), original.TASKS)
    ax.set_yticks(range(len(scores)), [original.DISPLAY_NAMES.get(m, m) for m in scores.index])
    ax.set_title('Recomputed task-local normalized average-rank scores')
    for i, j in zip(*np.where(np.isnan(matrix))): ax.text(j, i, 'NA', ha='center', va='center', color='gray')
    fig.colorbar(im, ax=ax, label='Normalized score')
    fig.tight_layout()
    for suffix in ['png', 'pdf', 'svg']: fig.savefig(output / f'cross_task_scores.{suffix}', dpi=200)
    plt.close(fig)
    union = ap[ap.reference == 'Union'].sort_values('AP')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = union.AP.to_numpy() * 100
    ax.barh(union.model, values, color='#327A94')
    errors = np.vstack([values-union.AP_CI_lower.to_numpy()*100, union.AP_CI_upper.to_numpy()*100-values])
    ax.errorbar(values, range(len(union)), xerr=errors, fmt='none', color='black', capsize=2)
    ax.axvline(438/163280*100, linestyle='--', color='gray', label='Reference prevalence')
    ax.set_xlabel('Pre-motif Union AP (%)')
    ax.set_title('Pre-motif AP and recomputed TF-cluster bootstrap intervals')
    ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ['png', 'pdf', 'svg']: fig.savefig(output / f'T5_Union_AP.{suffix}', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/reproduction.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduced')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    data = ROOT / config['data_directory']
    output = args.output.resolve()
    require(not output.is_relative_to(data.resolve()), 'Output must not overwrite bundled evidence')
    output.mkdir(parents=True, exist_ok=True)
    scores, contexts = rankings(data, output, config)
    ap = task5(data, output, config)
    plots(scores, ap, output)
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(data.rglob('*')) if p.is_file() and 'local' not in p.parts}
    report = dict(status='passed', task_context_records=contexts, S15_checked=True, S16_checked=True,
                  task5_AP_estimates_checked=len(ap), task5_bootstrap_intervals_recomputed=len(ap),
                  task5_bootstrap_values_recomputed=len(ap)*config['task5_bootstrap_replicates'],
                  task5_reference_label_vectors_reconstructed=3, S17_AUROC_checked=True,
                  max_bootstrap_AP_absolute_error=float(ap.bootstrap_AP_max_absolute_error.max()),
                  max_AP_absolute_error=float(ap.AP_absolute_error.max()),
                  python=platform.python_version(), platform=platform.platform(),
                  package_versions={n: importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn','matplotlib']},
                  input_sha256=hashes,
                  scope='Summary-table rankings; AP/AUROC from scores; reference-label reconstruction; 66000 bootstrap AP evaluations from scores and supplied TF multiplicities. No original training or upstream GRN rerun.')
    (output / 'run_manifest.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'PASS: S15, S16, AP/AUROC, reference labels and 66000 TF-bootstrap AP values. Output: {output}')


if __name__ == '__main__':
    main()
