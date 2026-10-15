"""Verify published aggregates; does not rerun training or patient bootstrap."""
import csv
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    manifest = json.loads((root / 'aggregate_manifest.json').read_text(encoding='utf-8'))
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.relative_to(root).parts}
    require(actual == set(manifest) | {'aggregate_manifest.json'}, 'Unexpected or missing files')
    for name, expected in manifest.items():
        require(hashlib.sha256((root / name).read_bytes()).hexdigest() == expected,
                f'Content hash mismatch: {name}')

    def rows(name):
        with (root / 'assessment' / name).open(encoding='utf-8', newline='') as handle:
            return list(csv.DictReader(handle))

    summaries = rows('model_summary.csv')
    models = {r['model']: r for r in summaries}
    require(len(models) == len(summaries) == 8, 'Expected eight distinct reported estimators')
    for row in summaries:
        for metric in ('volume_mae', 'setting_mae'):
            value = float(row[metric])
            require(math.isfinite(value) and value >= 0, f'Invalid {metric}')
    effects = rows('paired_effects.csv')
    for row in effects:
        metric = row['metric']
        expected = float(models[row['treatment']][metric]) - float(models[row['control']][metric])
        require(math.isclose(expected, float(row['delta_new_minus_control']), abs_tol=1e-8),
                'Paired point estimate disagrees with summary')
        require(float(row['ci_low']) <= float(row['ci_high']), 'Reversed interval')
        require(int(row['replicates']) == 2000, 'Unexpected bootstrap budget')
        require(int(row['paired_patients']) == (335 if metric == 'volume_mae' else 296),
                'Unexpected paired denominator')
    comparison = next(r for r in effects if r['treatment'] == 'physiology_only_hgb'
                      and r['control'] == 'observation_only_hgb' and r['metric'] == 'volume_mae')
    require(float(comparison['ci_low']) < 0 < float(comparison['ci_high']),
            'Report conclusion about independent physiology increment needs updating')
    inference = json.loads((root / 'trained_inference_audit.json').read_text(encoding='utf-8'))
    checks = inference['trained_model_intervention_checks']
    require(len(checks) == 5 and inference['requests'] == 9, 'Unexpected recorded intervention scope')
    require(all(c['maximum_prediction_difference_after_fluid_intervention'] == 0 for c in checks),
            'Recorded fluid intervention changed predictions')
    require(not inference['target_fields_passed_to_prediction'], 'Recorded target input violation')
    print(f'PASS: {len(manifest)} file hashes, {len(models)} estimators, '
          f'{len(effects)} paired aggregate comparisons; recorded intervention audit consistent.')
    print('This verifies published evidence consistency, not patient-level metrics or training reproduction.')


if __name__ == '__main__':
    verify(Path(__file__).resolve().parent)
