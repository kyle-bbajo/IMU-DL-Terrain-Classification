#!/bin/bash
# scripts/setup.sh — 본실험 환경 초기화 (g6.4xlarge)
set -e
REPO=$(dirname "$(dirname "$(realpath "$0")")")
VENV=/home/ubuntu/project/.venv
H5=/home/ubuntu/project/data/processed/batches/dataset.h5

echo "=== IMU-DL-Terrain-Classification 환경 초기화 ==="
echo "REPO: $REPO"

# 1) data 디렉토리 준비
mkdir -p $REPO/data/cache $REPO/results $REPO/logs
echo "[1] 디렉토리 완료"

# 2) HDF5 링크
if [ -f "$H5" ]; then
    ln -sf $H5 $REPO/data/dataset.h5
    echo "[2] HDF5 링크: $H5 ($(du -h $H5 | cut -f1))"
else
    echo "[2] 경고: HDF5 없음 $H5"
fi

# 3) Python 환경
echo "[3] Python 환경"
$VENV/bin/python -c "
import torch, numpy, sklearn
print(f'    torch={torch.__version__}  cuda={torch.cuda.is_available()}')
print(f'    numpy={numpy.__version__}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'    GPU: {props.name}  {props.total_memory//1024**3}GB')
"

# 4) GPU
echo "[4] GPU"
nvidia-smi --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader 2>/dev/null | awk '{print "    " $0}' || true

# 5) 디스크
echo "[5] 디스크"
df -h / /data 2>/dev/null | tail -n +2 | awk '{print "    " $0}'

echo ""
echo "=== 실행 방법 ==="
echo "  # 빠른 테스트 (5분)"
echo "  $VENV/bin/python -m src.experiments.run_all --fast"
echo ""
echo "  # 전체 실험 (기본 피처 조합, ~8시간)"
echo "  nohup $VENV/bin/python -m src.experiments.run_all > logs/run.log 2>&1 &"
echo ""
echo "  # 전체 실험 (피처 조합 16가지, ~20시간)"
echo "  nohup $VENV/bin/python -m src.experiments.run_all --all_feat_combos > logs/run_all.log 2>&1 &"
echo ""
echo "  # 모니터링"
echo "  tail -f logs/run.log"
