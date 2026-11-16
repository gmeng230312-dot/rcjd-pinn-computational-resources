# -*- coding: utf-8 -*-
"""Compare replay results with the archived numerical references."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
errors=[]
replay=ROOT/'verification_results/rcjd_replay.json'
if not replay.is_file():errors.append('RCJD replay output is missing')
else:
    d=json.loads(replay.read_text(encoding='utf-8'))
    if d.get('status')!='PASS' or d.get('samples')!=300:errors.append('RCJD replay check failed')
prior=ROOT/'outputs/posterior_replay/tables/step7v4_peak_reliability_table.csv'
reference=ROOT/'verification_reference/step7v4_peak_reliability_table_official.csv'
if not prior.exists():errors.append('Step7 posterior table is missing')
else:
    a=pd.read_csv(prior);b=pd.read_csv(reference)
    if a.shape!=b.shape or a.shape[0]!=9:errors.append('Step7 table shape differs from archived 9 rows')
    elif 'pf_posterior' not in a.columns:errors.append('Step7 probability columns unavailable')
    else:
        for col in ['threshold_m','pf_prior','pf_posterior']:
            if not np.allclose(a[col].values.astype(float),b[col].values.astype(float),rtol=0,atol=1e-10,equal_nan=True):
                errors.append('Step7 archival check differs: '+col)
summaries=[]
for p in (ROOT/'outputs/inversion').glob('Step6V2B_ExpandedBenchmark_*/step6v2b_expanded_inversion_summary.json'):
    try:
        d=json.loads(p.read_text(encoding='utf-8'))
        if d.get('n_cases')==30:summaries.append((p.stat().st_mtime,p,d))
    except Exception:pass
if not summaries:errors.append('Step6 complete 30-case summary is missing')
else:
    d=max(summaries)[2]
    if d.get('n_modes')!=8 or d.get('obs_count')!=494:errors.append('Step6 setup mismatch')
    if abs(float(d.get('mean_inv_RMSE_mm',99))-0.00531169929919684)>1e-4:
        errors.append('Step6 mean inversion RMSE differs from archived value')
    if abs(float(d.get('mean_initial_RMSE_mm',99))-0.014617367230679261)>1e-4:
        errors.append('Step6 initial RMSE differs from archived value')
output={'status':'PASS' if not errors else 'HOLD', 'checks':{
       'rcjd_300_sample_replay':replay.is_file(),
       'posterior_9_row_probability':prior.exists(),
       'inversion_30_cases':bool(summaries)},'errors':errors}
(ROOT/'verification_results').mkdir(exist_ok=True)
(ROOT/'verification_results/final_gate.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
print(json.dumps(output,indent=2))
if errors:sys.exit(2)
