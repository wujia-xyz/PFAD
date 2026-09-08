"""Produce strictly nested scores before registration-policy selection."""
from pathlib import Path
import argparse
import json
import hashlib
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'src/pfad/core.py'
sys.path.insert(0, str(ROOT / 'src'))
from pfad import core as M
RESULT = ROOT / 'results/first_arrival_refinement/rebuild_v2'
OUT = ROOT / 'artifacts/pfad_registration/nested_retention_v1/inner_scores'
CONFIG = ROOT / 'configs/pfad/registration/nested_retention_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    device = M.choose_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    specimens = [f'specimen{i:02}' for i in range(1, 15)]
    paths = {(s, a): M.discover_case_archive(RESULT / 'features', s, a)
             for s in specimens for a in M.ANATOMIES}
    max_sources = 0
    for p in paths.values():
        with np.load(p) as z:
            max_sources = max(max_sources, z['source_evidence'].shape[1])
    print('Device', str(device), 'torch', torch.__version__, flush=True)
    for heldout in specimens:
        dest = OUT / (heldout + '.npz')
        ledger = OUT / (heldout + '.json')
        if ledger.exists():
            d = json.loads(ledger.read_text())
            assert d['protocol_sha256'] == sha(CONFIG)
            assert d['model_source_sha256'] == sha(SOURCE)
            assert d['runner_sha256'] == sha(__file__)
            assert d['scores_sha256'] == sha(dest)
            print(heldout, 'reused', flush=True)
            continue
        start = time.perf_counter()
        training_specimens = [s for s in specimens if s != heldout]
        cases = [M.load_student_case(paths[(s, a)], max_sources,
                    M.discover_case_archive(RESULT / 'teacher_labels', s, a))
                 for s in training_specimens for a in M.ANATOMIES]
        scores, reports, checkpoints = {}, [], {}
        for fold in range(3):
            validation = set(training_specimens[fold::3])
            fitting = [c for c in cases if c['specimen'] not in validation]
            prediction = [c for c in cases if c['specimen'] in validation]
            train_ids = sorted({c['specimen'] for c in fitting})
            assert heldout not in train_ids and heldout not in validation
            assert not set(train_ids).intersection(validation)
            model, history = M.train_sweep_set_selector(fitting, epochs=12, batch_size=8192,
                learning_rate=.001, seed=17, device=device,
                student_variant='attentive_set', teacher_target='soft')
            model.eval()
            for case in prediction:
                prob = M.predict_sweep_set_selector(model, case, batch_size=8192, device=device)
                scores[case['specimen'] + '_' + case['anatomy']] = prob[:, 0] * prob[:, 1]
            checkpoints[str(fold)] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            reports.append(dict(fold=fold, training_specimens=train_ids,
                validation_specimens=sorted(validation), loss_history=history))
            del model
            torch.cuda.empty_cache()
        assert len(scores) == 39 and not any(k.startswith(heldout + '_') for k in scores)
        np.savez_compressed(dest, **scores)
        checkpoint = OUT / (heldout + '_models.pt')
        torch.save(checkpoints, checkpoint)
        result = dict(status='complete', heldout=heldout, training_specimens=training_specimens,
            folds=reports, scores_sha256=sha(dest), models_sha256=sha(checkpoint),
            protocol_sha256=sha(CONFIG), model_source_sha256=sha(SOURCE), runner_sha256=sha(__file__),
            input_sha256={str(p.relative_to(ROOT)): sha(p) for c in cases
                          for p in (c['feature_path'], c['label_path'])},
            cases=len(scores), seconds=time.perf_counter() - start)
        ledger.write_text(json.dumps(result, indent=2))
        print(heldout, 'complete', round(result['seconds'], 2), flush=True)
    print('COMPLETE: 42 inner fits, 546 cross-fit case predictions', flush=True)


if __name__ == '__main__':
    main()
