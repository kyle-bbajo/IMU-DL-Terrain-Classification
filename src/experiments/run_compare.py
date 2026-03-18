# -*- coding: utf-8 -*-
"""
src/experiments/run_compare.py — 모델 × 입력 × 피처 비교 실험

실험:
  5모델 × (raw / feat / raw_feat) × 피처 조합
  5-fold CV (Subject-wise StratifiedGroupKFold)

실행:
  python -m src.experiments.run_compare
  python -m src.experiments.run_compare --fast
  python -m src.experiments.run_compare --all_feat_combos
  python -m src.experiments.run_compare --input_mode raw_feat --models FusionNet
"""
from __future__ import annotations
import sys, argparse, gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.utils         import CFG, log, seed_everything, save_json, Timer, make_result_dir, H5_PATH, CACHE_DIR
from src.models        import ALL_MODELS
from src.features      import all_feat_type_combos, combo_name
from src.train.train_cv import run_kfold

# H5Data (기존 파일럿 코드에서 가져옴 — 향후 src/data/h5data.py로 이동 예정)
sys.path.insert(0, str(ROOT / "src"))
try:
    from data.h5data import H5Data
    from data.channel_info import build_branch_idx, get_foot_accel_idx
    from features.extractor import batch_extract
except ImportError:
    # 파일럿 코드 폴백
    PILOT = Path("/home/ubuntu/project/repo/src")
    sys.path.insert(0, str(PILOT))
    from train_common    import H5Data
    from channel_groups  import build_branch_idx, get_foot_accel_idx
    from features        import batch_extract


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models",          type=str,   default=",".join(ALL_MODELS))
    p.add_argument("--input_mode",      type=str,   default="all",
                   help="all | raw | feat | raw_feat")
    p.add_argument("--all_feat_combos", action="store_true",
                   help="피처 타입 모든 조합 2^4=16가지")
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
        args.kfold = args.kfold or 2
        args.epochs = args.epochs or 5
        args.early_stop = args.early_stop or 3

    seed_everything(args.seed)
    out = make_result_dir("compare")

    log("=" * 68)
    log(f"  run_compare.py  →  {out}")
    log(f"  models={args.models}  input={args.input_mode}")
    log(f"  kfold={args.kfold or CFG.kfold}  epochs={args.epochs or CFG.epochs}")
    log("=" * 68)

    # 데이터
    h5      = H5Data(str(H5_PATH))
    le      = LabelEncoder()
    y_all   = le.fit_transform(h5.y_raw).astype(np.int64)
    groups  = h5.subj_id
    X_all   = h5.X
    n_cls   = len(np.unique(y_all))
    branch_idx, branch_ch = build_branch_idx(h5.channels)
    foot_idx = get_foot_accel_idx(h5.channels)
    log(f"  N={len(y_all)}  피험자={len(np.unique(groups))}  클래스={n_cls}")

    # 피처 추출 (캐시)
    cache_path = CACHE_DIR / f"feat_seed{args.seed}.npy"
    if cache_path.exists() and not args.no_feat_cache:
        log(f"  ★ 피처 캐시 로드")
        feat_all = np.load(str(cache_path))
    else:
        log("  피처 추출 중...")
        with Timer() as t:
            feat_all = batch_extract(X_all, foot_idx, CFG.sample_rate,
                                     h5_path=str(H5_PATH))
        np.save(str(cache_path), feat_all)
        log(f"  완료 {feat_all.shape} ({t})")

    # 조합 구성
    if args.all_feat_combos:
        feat_combos = all_feat_type_combos(args.include_context)
    else:
        base = ["TIME","FREQ","GAIT","TERRAIN"]
        feat_combos = [[t] for t in base] + [base]

    input_modes = ["raw","feat","raw_feat"] if args.input_mode == "all" else [args.input_mode]
    model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    log(f"\n  {len(model_names)}모델 × {len(input_modes)}입력 × {len(feat_combos)}피처 = "
        f"{len(model_names)*len(input_modes)*len(feat_combos)}가지")

    all_results = []
    with Timer() as total:
        for model in model_names:
            for mode in input_modes:
                for fi, combo in enumerate(feat_combos):
                    if mode == "raw" and fi > 0: continue  # raw는 피처 무관하므로 1번만
                    name = f"{model}__{mode}__{combo_name(combo)}"
                    r = run_kfold(
                        exp_name=name, model_name=model, input_mode=mode,
                        feat_combo=combo if mode != "raw" else [],
                        X_all=X_all, feat_all=feat_all,
                        y_all=y_all, groups=groups,
                        branch_idx=branch_idx, branch_ch=branch_ch,
                        n_classes=n_cls, out_dir=out,
                        epochs=args.epochs, early_stop=args.early_stop,
                        kfold=args.kfold, seed=args.seed,
                    )
                    all_results.append(r)
                    gc.collect()

    save_json({"experiment":"compare","results":all_results,"total_time":str(total)},
              out / "summary_compare.json")

    log(f"\n  상위 10개 (F1)")
    for r in sorted(all_results, key=lambda x: x["macro_f1"], reverse=True)[:10]:
        log(f"  {r['exp_name']:<55}  Acc={r['acc']:.4f}  F1={r['macro_f1']:.4f}")
    log(f"\n  ★ 완료  {total}  →  {out}")
    h5.close()


if __name__ == "__main__":
    main()
