# IMU-DL-Terrain-Classification

IMU 기반 딥러닝 지면 분류 — 9센서 · 6클래스 · 50명 · 68,938 스텝

## 프로젝트 구조

```
IMU-DL-Terrain-Classification/
  src/
    models/         FlatCNN, BranchCNN, ResNet1D, ResNetTCN, FusionNet
    features/       피처 타입별 인덱스 맵 & 조합 생성 (builder.py)
    data/           Dataset, DataLoader, Split (5-fold / LOSO)
    train/          Trainer, Losses, train_cv
    experiments/    run_compare, run_feature_ablation, run_sensor_ablation, run_loso, run_all
    utils/          config, logger, seed, metrics
  data/             → HDF5 symlink (gitignore)
  results/          → 실험 결과 (날짜_실험명/)
  logs/
  scripts/setup.sh
```

## 모델 구성

| 이름 | 입력 | 역할 |
|------|------|------|
| FlatCNN | 로우 신호 (54ch) | 베이스라인 |
| BranchCNN | 로우 신호 → 5그룹 브랜치 + CBAM | 비교 모델 1 |
| ResNet1D | 로우 신호 → 잔차 블록 | 비교 모델 2 |
| ResNetTCN | 로우 신호 → ResNet + Dilated TCN | 비교 모델 3 |
| **FusionNet** | 로우 신호 + 도메인 피처(390차원) | **제안 모델** |

## 실험 구성

- **모델 비교**: 5모델 × (raw / feat / raw+feat) × 피처 조합 × 5-fold
- **피처 ablation**: 5가지 타입(TIME/FREQ/GAIT/TERRAIN/CONTEXT) 16가지 조합
- **센서 ablation**: Foot → +Shank → +Thigh → +Hand → +Pelvis → 전체
- **LOSO**: 피험자 단위 leave-one-out 일반화 검증

## 실행

```bash
# 환경 초기화
bash scripts/setup.sh

# 빠른 테스트
python -m src.experiments.run_all --fast

# 전체 실험
nohup python -m src.experiments.run_all --all_feat_combos > logs/run.log 2>&1 &
```

## 데이터

- 센서: Noraxon IMU × 9 (54ch, 200Hz)
- 지면: C1 미끄러운 / C2 오르막 / C3 내리막 / C4 흙길 / C5 잔디 / C6 평지
- 피험자: 50명 / 스텝: 68,938개

## 서버 환경

- AWS g6.4xlarge (NVIDIA L4 24GB, vCPU 16, RAM 64GB, NVMe 419GB)
- Python 3.12, PyTorch 2.x, BF16 AMP
