# -*- coding: utf-8 -*-
"""Read-only presence, metadata and numerical consistency checks."""
from pathlib import Path
import json,hashlib,ast
import numpy as np
ROOT=Path(__file__).resolve().parent
must=[ROOT/'05_RCJD_Artifacts/best_rcjd_model.pt',ROOT/'05_RCJD_Artifacts/test_prediction_curves.npz',
      ROOT/'02_Inversion/data/step6_latent_prior_modes.npz',
      ROOT/'03_Posterior/data/step7_ready_nonparametric_posterior_case_median.npz']
must += [ROOT/f'06_Dataset/pinn_node_dataset_{s}.npz' for s in ('train','val','test')]
must += list((ROOT/'01_RCJD_PINN').glob('*.py'))+list((ROOT/'02_Inversion').glob('*.py'))
for f in must:
    if not f.is_file():raise FileNotFoundError(f)
with np.load(ROOT/'03_Posterior/data/step7_ready_nonparametric_posterior_case_median.npz',allow_pickle=False) as z:
    assert z['w_prior'].shape==(2000,500)
    assert z['faulting_prior'].shape==(2000,49)
    assert z['x_joint_m'][0]==1.2 and abs(z['x_joint_m'][-1]-58.8)<1e-8
    assert np.array_equal(z['left_node_indices'],np.arange(9,500,10))
    assert np.array_equal(z['right_node_indices'],np.arange(10,500,10))
    assert abs(float(z['posterior_ess'])-200.163908949)<1e-5
    assert abs(np.sum(z['posterior_weights'])-1.0)<1e-10
    w=z['w_prior'];j=np.abs(w[:,9::10][:,:49]-w[:,10::10][:,:49]);assert np.array_equal(j,z['faulting_prior'])
for p in ROOT.rglob('*.py'):
    ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
print('PREFLIGHT_PASS | response=2000x500 | joints=49 | ESS=200.1639 | python_syntax=PASS')
if __name__=='__main__':pass
