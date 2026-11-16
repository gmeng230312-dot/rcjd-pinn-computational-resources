# -*- coding: utf-8 -*-
"""Reload the archived RCJD-PINN checkpoint and compare all 300 test predictions."""
import json,sys,hashlib,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'01_RCJD_PINN'))
import step4d_rcjd_pinn_forward_config as cfg
from step4d_common import (build_model, namespace_from_config_dict, load_checkpoint,
                            predict_split, load_split, regression_metrics)

def main():
    out=ROOT/'verification_results';out.mkdir(exist_ok=True)
    p=ROOT/'05_RCJD_Artifacts/best_rcjd_model.pt'
    ckpt=load_checkpoint(p,map_location='cpu')  # bundled trusted artifact
    model=build_model(namespace_from_config_dict(cfg,ckpt['config']))
    model.load_state_dict(ckpt['model_state_dict'])
    test=load_split(cfg.TEST_NPZ,'test',cfg)
    result=predict_split(model,test,np.asarray(ckpt['x_mean'],dtype=np.float32),
                         np.asarray(ckpt['x_std'],dtype=np.float32),
                         np.asarray(ckpt['y_mean'],dtype=np.float32),
                         np.asarray(ckpt['y_std'],dtype=np.float32),cfg,device='cpu')
    with np.load(ROOT/'05_RCJD_Artifacts/test_prediction_curves.npz',allow_pickle=False) as z:
        y_ref=z['y_pred_m']
        id_ref=z['global_sample_indices'].astype('int64')
    assert np.array_equal(result['global_idx'],id_ref), 'Test sample order does not match archive'
    assert y_ref.shape==result['y_pred_m'].shape==(300,500)
    diff=np.asarray(result['y_pred_m'],dtype='float64')-y_ref
    d_rmse=np.sqrt(np.mean((diff*1000.0)**2))
    d_max=float(np.max(np.abs(diff))*1000)
    metrics=regression_metrics(result['y_true_m'],result['y_pred_m'],1000.0)
    passed=bool(d_rmse<1e-5 and d_max<1e-4)
    payload={'status':'PASS' if passed else 'FAIL','samples':300,
             'diff_rmse_mm':float(d_rmse),'diff_max_abs_mm':d_max,
             'test_metrics':metrics,'checkpoint_epoch':int(ckpt['epoch']),
             'model_params':sum(x.numel() for x in model.parameters())}
    (out/'rcjd_replay.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(payload,ensure_ascii=False,indent=2))
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()
