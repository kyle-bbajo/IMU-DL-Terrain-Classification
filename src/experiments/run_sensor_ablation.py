# -*- coding: utf-8 -*-
"""
src/experiments/run_sensor_ablation.py — 센서 그룹 누적 추가 ablation

실험:
  Foot만 → +Shank → +Thigh → +Hand → +Pelvis → +Terrain
  → "어떤 센서 조합이 최적인가?"

실행:
  python -m src.experiments.run_sensor_ablation
"""
from __future__ import annotations
import sys, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils          import CFG, log, seed_everything, save_json, Timer, make_result_dir, H5_PATH, CACHE_DIR
from src.features       import sensor_cumulative_combos, get_sensor_mask, combo_name
from src.train.train_cv import run_kfold

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
    p.add_argument("--model",         type=str,  default="FusionNet")
    p.add_argument("--input_mode",    type=str,  default="raw_feat")
    p.add_argument("--kfold",         type=int,  default=None)
    p.add_argument("--epochs",        type=int,  default=None)
    p.add_argument("--early_stop",    type=int,  default=None)
    p.add_argument("--seed",          type=int,  default=42)
    p.add_argument("--no_feat_cache", action="store_true")
    p.add_argument("--fast",          action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.kfold = args.kfold or 2; args.epochs = args.epochs or 5

    seed_everything(args.seed)
    out = make_result_dir("sensor_ablation")

    log("=" * 68)
    log(f"  run_sensor_ablation.py  model={args.model}")
    log("=" * 68)

    h5      = H5Data(str(H5_PATH))
    le      = LabelEncoder()
    y_all   = le.fit_transform(h5.y_raw).astype(np.int64)
    groups  = h5.subj_id; X_all = h5.X
    n_cls   = len(np.unique(y_all))
    branch_idx_full, branch_ch_full = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)

    cache_path = CACHE_DIR / f"feat_seed{args.seed}.npy"
    if cache_path.exists() and not args.no_feat_cache:
        feat_all = np.load(str(cache_path))
    else:
        with Timer() as t:
            feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate, h5_path=str(H5_PATH))
        np.save(str(cache_path), feat_all)

    sensor_combos = sensor_cumulative_combos()
    all_results   = []

    with Timer() as total:
        for sensors in sensor_combos:
            # 해당 센서의 피처 인덱스만 마스킹
            fidx = get_sensor_mask(sensors, n_features=feat_all.shape[1])
            feat_masked = feat_all[:, fidx]

            # branch_ch도 해당 센서만
            b_ch  = {nm: ch for nm, ch in branch_ch_full.items() if nm in sensors}
            b_idx = {nm: idx for nm, idx in branch_idx_full.items() if nm in sensors}

            name = f"{args.model}__sensor__{combo_name(sensors)}"
            r = run_kfold(
                exp_name=name, model_name=args.model, input_mode=args.input_mode,
                feat_combo=[],  # 피처는 센서 마스크로 대체
                X_all=X_all, feat_all=feat_masked,
                y_all=y_all, groups=groups,
                branch_idx=b_idx, branch_ch=b_ch,
                n_classes=n_cls, out_dir=out,
                epochs=args.epochs, early_stop=args.early_stop,
                kfold=args.kfold, seed=args.seed,
            )
            all_results.append(r)
            log(f"  센서: {sensors}  →  F1={r['macro_f1']:.4f}")
            gc.collect()

    save_json({"experiment":"sensor_ablation","model":args.model,
               "results":all_results,"total_time":str(total)},
              out / "summary_sensor_ablation.json")

    log(f"\n  센서 누적 추가 결과:")
    for r in all_results:
        log(f"  {str(r['feat_combo']):<50}  F1={r['macro_f1']:.4f}")
    log(f"\n  ★ 완료  {total}")
    h5.close()


if __name__ == "__main__":
    main()
