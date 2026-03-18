#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/experiments/run_all.py — 전체 실험 순차 실행

Phase 1: 모델 비교       (run_compare.py)
Phase 2: 피처 ablation   (run_feature_ablation.py)
Phase 3: 센서 ablation   (run_sensor_ablation.py)
Phase 4: LOSO 검증       (run_loso.py)

실행:
  python -m src.experiments.run_all              # 전체
  python -m src.experiments.run_all --phase 1    # Phase 1만
  python -m src.experiments.run_all --fast       # 빠른 테스트
  python -m src.experiments.run_all --all_feat_combos  # 피처 전체 조합
"""
from __future__ import annotations
import sys, subprocess, argparse, time
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[3]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
PY      = sys.executable


def ts(): return time.strftime("%H:%M:%S")


def run(name: str, cmd: list[str], log_path: Path) -> bool:
    print(f"\n[{ts()}] ▶ {name}")
    print(f"  LOG: {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    ok = ret.returncode == 0
    print(f"  {'✅ 완료' if ok else '❌ 실패'}  (code={ret.returncode})")
    if not ok:
        lines = log_path.read_text(encoding="utf-8").splitlines()
        for l in lines[-15:]: print(f"  {l}")
    return ok


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase",           type=int, default=0,
                   help="0=전체 1=compare 2=feat_ablation 3=sensor_ablation 4=loso")
    p.add_argument("--models",          type=str, default="FlatCNN,BranchCNN,ResNet1D,ResNetTCN,FusionNet")
    p.add_argument("--all_feat_combos", action="store_true")
    p.add_argument("--include_context", action="store_true")
    p.add_argument("--kfold",           type=int, default=5)
    p.add_argument("--epochs",          type=int, default=80)
    p.add_argument("--early_stop",      type=int, default=15)
    p.add_argument("--fast",            action="store_true")
    return p.parse_args()


def main():
    args  = parse_args()
    ts_id = time.strftime("%Y%m%d_%H%M%S")
    fast  = ["--fast"] if args.fast else []
    ep    = ["--epochs",     str(5  if args.fast else args.epochs)]
    kf    = ["--kfold",      str(2  if args.fast else args.kfold)]
    es    = ["--early_stop", str(3  if args.fast else args.early_stop)]

    status = {}

    print("=" * 68)
    print(f"  IMU-DL-Terrain-Classification  본실험 파이프라인")
    print(f"  {ts_id}  fast={args.fast}")
    print("=" * 68)

    # Phase 1: 모델 비교
    if args.phase in (0, 1):
        cmd = [PY, "-m", "src.experiments.run_compare",
               "--models", args.models] + ep + kf + es
        if args.all_feat_combos: cmd.append("--all_feat_combos")
        if args.include_context: cmd.append("--include_context")
        if args.fast: cmd.append("--fast")
        status["compare"] = run("Phase 1: 모델 비교", cmd, LOG_DIR/f"compare_{ts_id}.log")

    # Phase 2: 피처 ablation
    if args.phase in (0, 2):
        cmd = [PY, "-m", "src.experiments.run_feature_ablation",
               "--model", "FusionNet", "--all_combos"] + ep + kf + es
        if args.fast: cmd.append("--fast")
        status["feat_ablation"] = run("Phase 2: 피처 ablation", cmd, LOG_DIR/f"feat_{ts_id}.log")

    # Phase 3: 센서 ablation
    if args.phase in (0, 3):
        cmd = [PY, "-m", "src.experiments.run_sensor_ablation",
               "--model", "FusionNet"] + ep + kf + es
        if args.fast: cmd.append("--fast")
        status["sensor_ablation"] = run("Phase 3: 센서 ablation", cmd, LOG_DIR/f"sensor_{ts_id}.log")

    # Phase 4: LOSO
    if args.phase in (0, 4):
        cmd = [PY, "-m", "src.experiments.run_loso",
               "--models", "FusionNet,BranchCNN,ResNetTCN"] + ep + es
        if args.fast: cmd += ["--fast", "--n_subjects", "10"]
        status["loso"] = run("Phase 4: LOSO 검증", cmd, LOG_DIR/f"loso_{ts_id}.log")

    print(f"\n{'='*68}")
    print(f"  완료  {time.strftime('%H:%M:%S')}")
    for phase, ok in status.items():
        print(f"  {'✅' if ok else '❌'} {phase}")
    print(f"  결과: {ROOT/'results'}")
    print(f"{'='*68}")


if __name__ == "__main__":
    main()
