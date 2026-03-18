# -*- coding: utf-8 -*-
"""
src/experiments/run_feature_ablation.py — 피처 타입 조합 ablation

실험:
  FusionNet 고정, 피처 조합 모든 경우의 수 (2^4=16 or 2^5=32)
  → "어떤 피처 타입이 성능에 기여하는가?"

실행:
  python -m src.experiments.run_feature_ablation
  python -m src.experiments.run_feature_ablation --all_combos --include_context
"""
from __future__ import annotations
import sys, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils         import CFG, log, seed_everything, save_json, Timer, make_result_dir, H5_PATH, CACHE_DIR
from src.features      import all_feat_type_combos, combo_name, FEAT_TYPES_NO_CTX
from src.train.train_cv import run_kfold

sys.path.insert(0, str(ROOT / "src"))
try:
    from data.h5data       import H5Data
    from data.channel_info import build_branch_idx, get_foot_accel_idx
    from features.extractor import batch_extract
except ImportError:
    PILOT = Path("/home/ubuntu/project/repo/src")
    sys.path.insert(0, str(PILOT))
    from train_common   import H5Data
    from channel_groups import build_branch_idx, get_foot_accel_idx
    from features       import batch_extract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           type=str,   default="FusionNet")
    p.add_argument("--input_mode",      type=str,   default="raw_feat")
    p.add_argument("--all_combos",      action="store_true", help="2^4=16가지 전체")
    p.add_argument("--include_context", action="store_true")
    p.add_argument("--kfold",           type=int,   default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--early_stop",      type=int,   default=None)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--no_feat_cache",   action="store_true")
    p.add_argument("--fast",            action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.fast:
        args.kfold = args.kfold or 2; args.epochs = args.epochs or 5

    seed_everything(args.seed)
    out = make_result_dir("feature_ablation")

    log("=" * 68)
    log(f"  run_feature_ablation.py  model={args.model}  mode={args.input_mode}")
    log("=" * 68)

    # 데이터
    h5      = H5Data(str(H5_PATH))
    le      = LabelEncoder()
    y_all   = le.fit_transform(h5.y_raw).astype(np.int64)
    groups  = h5.subj_id; X_all = h5.X
    n_cls   = len(np.unique(y_all))
    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)

    cache_path = CACHE_DIR / f"feat_seed{args.seed}.npy"
    if cache_path.exists() and not args.no_feat_cache:
        feat_all = np.load(str(cache_path))
    else:
        with Timer() as t:
            feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate, h5_path=str(H5_PATH))
        np.save(str(cache_path), feat_all)
        log(f"  피처 추출 완료 {feat_all.shape} ({t})")

    # 피처 조합
    if args.all_combos:
        combos = all_feat_type_combos(args.include_context)
    else:
        # 기본: 타입별 단독(4) + 누적 추가(4) + 전체(1) = 9가지
        base = FEAT_TYPES_NO_CTX
        single = [[t] for t in base]
        cumul  = [base[:i+1] for i in range(len(base))]
        combos = single + [c for c in cumul if c not in single]

    # 로우만도 베이스라인으로 추가
    all_results = []
    with Timer() as total:
        # 로우만 베이스라인
        r = run_kfold("raw_baseline", args.model, "raw", [],
                      X_all, feat_all, y_all, groups, branch_idx, branch_ch,
                      n_cls, out, args.epochs, args.early_stop, args.kfold, args.seed)
        all_results.append(r)

        for combo in combos:
            name = f"{args.model}__{args.input_mode}__{combo_name(combo)}"
            r = run_kfold(name, args.model, args.input_mode, combo,
                          X_all, feat_all, y_all, groups, branch_idx, branch_ch,
                          n_cls, out, args.epochs, args.early_stop, args.kfold, args.seed)
            all_results.append(r)
            gc.collect()

    save_json({"experiment":"feature_ablation","model":args.model,
               "results":all_results,"total_time":str(total)},
              out / "summary_feature_ablation.json")

    log(f"\n  피처 조합별 결과 (F1 정렬)")
    for r in sorted(all_results, key=lambda x: x["macro_f1"], reverse=True):
        log(f"  {r['feat_combo']!s:<40}  F1={r['macro_f1']:.4f}  Acc={r['acc']:.4f}")
    log(f"\n  ★ 완료  {total}")
    h5.close()


if __name__ == "__main__":
    main()
