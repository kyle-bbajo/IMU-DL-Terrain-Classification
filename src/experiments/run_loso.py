# -*- coding: utf-8 -*-
"""
src/experiments/run_loso.py — LOSO 검증

실험:
  5모델 × (raw / raw_feat) LOSO
  → 새 피험자에 대한 일반화 성능

실행:
  python -m src.experiments.run_loso
  python -m src.experiments.run_loso --models FusionNet,BranchCNN
"""
from __future__ import annotations
import sys, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils          import CFG, log, seed_everything, save_json, Timer, make_result_dir, H5_PATH, CACHE_DIR
from src.models         import ALL_MODELS
from src.features       import combo_name
from src.train.train_cv import run_loso

sys.path.insert(0, str(ROOT / "src"))
try:
    from data.h5data        import H5Data
    from data.channel_info  import build_branch_idx, get_foot_accel_idx
    from features.extractor import batch_extract
except ImportError:
    PILOT = Path("/home/ubuntu/project/repo/src")
    sys.path.insert(0, str(PILOT))
    from train_common   import H5Data
    from channel_groups import build_branch_idx, get_foot_accel_idx
    from features       import batch_extract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",          type=str,  default=",".join(ALL_MODELS))
    p.add_argument("--input_mode",      type=str,  default="raw_feat")
    p.add_argument("--feat_combo",      type=str,  default="TIME,FREQ,GAIT,TERRAIN")
    p.add_argument("--epochs",          type=int,  default=None)
    p.add_argument("--early_stop",      type=int,  default=None)
    p.add_argument("--n_subjects",      type=int,  default=None)
    p.add_argument("--seed",            type=int,  default=42)
    p.add_argument("--no_feat_cache",   action="store_true")
    p.add_argument("--fast",            action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.epochs = args.epochs or 5; args.n_subjects = args.n_subjects or 10

    seed_everything(args.seed)
    out = make_result_dir("loso")

    log("=" * 68)
    log(f"  run_loso.py  models={args.models}  mode={args.input_mode}")
    log("=" * 68)

    h5      = H5Data(str(H5_PATH))
    le      = LabelEncoder()
    y_all   = le.fit_transform(h5.y_raw).astype(np.int64)
    groups  = h5.subj_id; X_all = h5.X

    if args.n_subjects:
        keep   = np.unique(groups)[:args.n_subjects]
        mask   = np.isin(groups, keep)
        X_all  = X_all[mask]; y_all = y_all[mask]; groups = groups[mask]

    n_cls  = len(np.unique(y_all))
    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)
    log(f"  N={len(y_all)}  피험자={len(np.unique(groups))}  클래스={n_cls}")

    cache_path = CACHE_DIR / f"feat_seed{args.seed}.npy"
    if cache_path.exists() and not args.no_feat_cache:
        feat_all = np.load(str(cache_path))
        if args.n_subjects: feat_all = feat_all[mask]
    else:
        with Timer() as t:
            feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate, h5_path=str(H5_PATH))
        np.save(str(cache_path), feat_all)
        log(f"  피처 추출 완료 ({t})")

    feat_combo  = [f.strip() for f in args.feat_combo.split(",") if f.strip()]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    all_results = []

    with Timer() as total:
        for model in model_names:
            for mode in (["raw","raw_feat"] if args.input_mode == "all" else [args.input_mode]):
                name = f"{model}__{mode}"
                r = run_loso(
                    exp_name=name, model_name=model, input_mode=mode,
                    feat_combo=feat_combo if mode != "raw" else [],
                    X_all=X_all, feat_all=feat_all,
                    y_all=y_all, groups=groups,
                    branch_idx=branch_idx, branch_ch=branch_ch,
                    n_classes=n_cls, out_dir=out,
                    epochs=args.epochs, early_stop=args.early_stop,
                )
                all_results.append(r)
                gc.collect()

    save_json({"experiment":"loso","results":all_results,"total_time":str(total)},
              out / "summary_loso.json")

    log(f"\n  LOSO 결과")
    for r in sorted(all_results, key=lambda x: x["macro_f1"], reverse=True):
        log(f"  {r['exp_name']:<30}  Acc={r['acc']:.4f}  F1={r['macro_f1']:.4f}")
    log(f"\n  ★ 완료  {total}")
    h5.close()


if __name__ == "__main__":
    main()
