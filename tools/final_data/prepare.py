"""Materialize versioned extraction inputs, never model runs or predictions."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def prepare(repository, destination):
    repository=Path(repository).resolve();destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=False)
    source=repository/'data/datasets/reconstruction/clock_corrected_v3_20261015'
    names=['raw_source_manifest.csv','patient_index.csv','final_v2_hourly.csv',
           'final_v2_event_decisions.csv','final_v2_lot_accounting.csv',
           'final_v2_integration_segments.csv','final_v2_volume_constraints.csv',
           'final_v2_oral_decisions.csv','final_v2_case_decisions.csv',
           'final_v2_text_case_decisions.csv','run_manifest.json','operative_phases.json']
    provenance=[]
    for name in names:
        p=source/name;shutil.copy2(p,destination/name)
        provenance.append(dict(path=name,source_path=str(p.relative_to(repository)),sha256=digest(p)))
    # These arrays carry previously extracted boundaries and historical cohort
    # identities only. No fitted weights, predictions, windows or model scores.
    p=repository/'data/datasets/frozen/inventory_state_v3_20261016/features.npz'
    with np.load(p,allow_pickle=False) as z:
        patient=pd.DataFrame({'patient_id':z['patient_ids'].astype(str),
          'historical_holdout':z['historical_holdout_mask'],
          'historical_development_fold':z['development_outer_fold'],
          'admission_hour':z['admission_hour'],
          'observation_end_lower_hour':z['observation_end_lower_hour'],
          'observation_end_upper_hour':z['observation_end_upper_hour'],
          'observation_end_status':z['observation_end_status'],
          'cross_patient_volume_prior_dependency':z['volume_patient_cohort_prior_dependency']})
    patient.to_csv(destination/'boundary_reference.csv',index=False)
    provenance.append(dict(path='boundary_reference.csv',source_path=str(p.relative_to(repository)),
                           source_sha256=digest(p),sha256=digest(destination/'boundary_reference.csv'),
                           selection='Only cohort/boundary/prior-dependency metadata; no experiment inputs or predictions'))
    p=repository/'data/archive/datasets/adapters/inventory_boundary_v2_20261016/boundary_event_ledger_private.csv'
    shutil.copy2(p,destination/'boundary_event_ledger.csv')
    provenance.append(dict(path='boundary_event_ledger.csv',source_path=str(p.relative_to(repository)),sha256=digest(p)))
    # Historical extraction recipes are evidence for how the supplied adjudicated
    # reference tables were made, not a claim this new driver recreates all past
    # clinical judgements from raw records alone.
    snapshots=destination/'reference_recipe';snapshots.mkdir()
    for p in (source/'code_snapshot').iterdir():
        if p.is_file():
            shutil.copy2(p,snapshots/p.name)
            provenance.append(dict(path=f'reference_recipe/{p.name}',source_path=str(p.relative_to(repository)),sha256=digest(p)))
    (destination/'input_manifest.json').write_text(json.dumps(dict(
      status='Frozen extraction-level inputs; contains private patient evidence',
      scope='449 original adjudicated source records plus416 baseline-admission eligible boundary references',
      reproduction='Final table assembly and new raw-source extraction reproducible from these inputs plus raw sources. Historical adjudication tables and recipe retained as supplied evidence, not independently clinically re-adjudicated.',
      files=provenance),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return destination
