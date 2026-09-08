"""Fixed-policy LOSO context ablation on cached PFAD data; no model selection.

The baseline and both ablations have identical dimensions, seed, training and
record-level retention. Held-out labels are read only after scores are saved.
Endpoints concern candidate selection, not corrected 3-D surface geometry.
"""
from pathlib import Path
import sys
import json
import time
import hashlib
import importlib.util
import argparse
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from pfad import core as M
RESULT = ROOT / 'results/first_arrival_refinement/rebuild_v2'
OUT = ROOT / 'artifacts/pfad_tmi_revision_20260905/context_ablation'
VARIANTS = ('full', 'no_anatomy', 'anatomy_only_context')


def variant_case(case, variant):
    out = dict(case)
    out['global'] = case['global'].copy()
    if variant == 'no_anatomy':
        out['global'][:, 16:19] = 0
    elif variant == 'anatomy_only_context':
        out['global'][:, :16] = 0
    return out


def main():
    global OUT, VARIANTS
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--full-only', action='store_true')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    if args.full_only:
        VARIANTS = ('full',)
        OUT = OUT.parent / f'seed_{args.seed}'
    torch.set_num_threads(4)
    device = M.choose_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'status': 'running', 'design': '14-specimen LOSO fixed-policy retrospective mechanism study',
                'variants': list(VARIANTS), 'seed': args.seed, 'epochs': 12, 'batch_size': 8192,
                'retention': .9, 'score': 'factor', 'selection_of_winner': False,
                'endpoints': ['joint_f1', 'joint_auroc', 'joint_ap', 'uncorrected_accuracy_mm'],
                'scope': 'cached released inputs; no external or end-to-end upstream validation',
                'implementation_sha256': hashlib.sha256((ROOT / 'scripts/run_first_arrival_refinement.py').read_bytes()).hexdigest()}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    specimens = [f'specimen{i:02}' for i in range(1, 15)]
    paths = {(s, a): M.discover_case_archive(RESULT / 'features', s, a) for s in specimens for a in M.ANATOMIES}
    max_sources = max(np.load(p)['source_evidence'].shape[1] for p in paths.values())
    cases = {key: M.load_student_case(path, max_sources) for key, path in paths.items()}
    print('Loaded unlabeled cases', len(cases), flush=True)
    records = []
    for specimen in specimens:
        train = [M.load_student_case(path, max_sources, M.discover_case_archive(RESULT / 'teacher_labels', s, a))
                 for (s, a), path in paths.items() if s != specimen]
        for variant in VARIANTS:
            dest = OUT / f'{specimen}_{variant}.json'
            if dest.exists():
                records.extend(json.loads(dest.read_text())['cases'])
                continue
            start = time.monotonic()
            model, history = M.train_sweep_set_selector([variant_case(c, variant) for c in train],
                epochs=12, batch_size=8192, learning_rate=.001, seed=args.seed, device=device,
                student_variant='attentive_set', teacher_target='soft')
            model.eval()
            fold_rows = []
            for anatomy in M.ANATOMIES:
                case = variant_case(cases[(specimen, anatomy)], variant)
                prob = M.predict_sweep_set_selector(model, case, batch_size=8192, device=device)
                score = prob[:, 0] * prob[:, 1]
                keep, _ = M.coverage_constrained_selector_mask(score, float('inf'), case['counts'], .9)
                prediction_path = OUT / f'{specimen}_{anatomy}_{variant}.npz'
                np.savez_compressed(prediction_path, score=score, keep=keep)
                # Labels enter the evaluation only after the prediction artifact is written.
                with np.load(M.discover_case_archive(RESULT / 'teacher_labels', specimen, anatomy)) as label:
                    target = label['joint_target'].astype(bool)
                    distance = label['target_surface_distance_mm']
                    row = M.binary_selection_metrics(target, keep)
                    row.update({'specimen': specimen, 'anatomy': anatomy, 'variant': variant,
                        'joint_f1': row.pop('fscore'), 'joint_auroc': float(roc_auc_score(target, score)),
                        'joint_ap': float(average_precision_score(target, score)),
                        'uncorrected_accuracy_mm': float(distance[keep].mean()),
                        'prediction_sha256': hashlib.sha256(prediction_path.read_bytes()).hexdigest()})
                if variant == 'full' and args.seed == 17 and (RESULT / 'selector_attentive_loocv/soft_factor90' / f'{specimen}_{anatomy}.npz').exists():
                    with np.load(RESULT / 'selector_attentive_loocv/soft_factor90' / f'{specimen}_{anatomy}.npz') as old:
                        old_score = old['selector_anatomy_probability'] * old['selector_surface_probability']
                        row['replay_score_max_abs_difference'] = float(np.max(np.abs(score-old_score)))
                        row['replay_selection_agreement'] = float(np.mean(keep == old['selection_keep']))
                fold_rows.append(row)
            (dest).write_text(json.dumps({'cases': fold_rows, 'loss_history': history,
                                         'seconds': time.monotonic()-start}, indent=2))
            records.extend(fold_rows)
            print(specimen, variant, 'seconds', round(time.monotonic()-start, 1),
                  'joint F1', round(np.mean([r['joint_f1'] for r in fold_rows]), 5), flush=True)
            del model
            torch.cuda.empty_cache()
        del train
    assert len(records) == 42 * len(VARIANTS)
    summary = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for metric in protocol['endpoints']:
            vals = [np.mean([r[metric] for r in records if r['specimen']==s and r['variant']==variant]) for s in specimens]
            summary[variant][metric] = {'mean': float(np.mean(vals)), 'specimen_values': vals}
    comparisons = {}
    for variant in VARIANTS[1:]:
        endpoints = {}
        for metric in protocol['endpoints']:
            ref = np.array(summary['full'][metric]['specimen_values'])
            alt = np.array(summary[variant][metric]['specimen_values'])
            endpoints[metric] = M.paired_endpoint_statistics(ref, alt,
                direction='lower' if metric.endswith('_mm') else 'higher',
                bootstrap_resamples=100000, bootstrap_seed=20260905)
        M.add_holm_adjustment(endpoints)
        comparisons[variant] = endpoints
    (OUT / 'summary.json').write_text(json.dumps({'status':'complete','summary':summary,'comparisons':comparisons,'cases':records}, indent=2))
    protocol['status'] = 'complete'
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('COMPLETE', json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
