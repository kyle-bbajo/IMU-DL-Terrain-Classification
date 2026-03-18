# IMU-DL-Terrain-Classification

IMU 센서 기반 딥러닝 지면 분류 연구

- **센서**: Noraxon IMU × 9 (Pelvis / Hand×2 / Thigh×2 / Shank×2 / Foot×2)
- **채널**: 54ch (Accel 3축 + Gyro 3축 × 9센서), 200Hz
- **지면**: 6종 (미끄러운 / 오르막 / 내리막 / 흙길 / 잔디 / 평지)
- **피험자**: 50명 / **스텝**: 68,938개
- **서버**: AWS g6.4xlarge (NVIDIA L4 24GB, vCPU 16, RAM 64GB, NVMe 419GB)

---

## 프로젝트 구조

```
IMU-DL-Terrain-Classification/
  src/
    models/         FlatCNN, BranchCNN, ResNet1D, ResNetTCN, FusionNet
    features/       피처 타입별 인덱스 맵 & 조합 생성
    data/           Dataset, DataLoader, 5-fold / LOSO 분할
    train/          Trainer, Losses, 학습 루프
    experiments/    실험 스크립트
    utils/          config, logger, seed, metrics
  tests/            단위 테스트 (test_sanity.py)
  data/             → HDF5 symlink (.gitignore)
  results/          → 날짜_실험명/ 구조로 자동 저장
  logs/
  scripts/setup.sh
```

---

## 모델 구성

| 이름 | 입력 | 역할 |
|------|------|------|
| **FlatCNN** | 로우 신호 54ch | 베이스라인 |
| **BranchCNN** | 로우 신호 → 5그룹 브랜치 + CBAM + CrossGroupAttn | 비교 모델 1 |
| **ResNet1D** | 로우 신호 → 잔차 블록 1D CNN | 비교 모델 2 |
| **ResNetTCN** | 로우 신호 → ResNet + Dilated TCN | 비교 모델 3 |
| **FusionNet** | 로우 신호 + 도메인 피처 390차원 융합 + 속성 보조헤드 | **제안 모델** |

---

## 피처 구성 (390차원)

| 그룹 | 차원 | 피처 타입 |
|------|------|----------|
| Pelvis | 30 | TIME, FREQ, GAIT |
| Hand | 38 | TIME, FREQ, GAIT |
| Thigh | 36 | TIME, FREQ, GAIT |
| Shank | 40 | TIME, GAIT |
| Foot | 88 | TIME, FREQ, GAIT |
| Terrain | 98 | TERRAIN (충격/진동/roughness/stiffness) |
| Context | 60 | CONTEXT (bout 컨텍스트) |

---

## 실험 구성

| 실험 | 내용 |
|------|------|
| **모델 비교** | 5모델 × (raw / feat / raw+feat) × 피처조합 × 5-fold |
| **피처 ablation** | TIME/FREQ/GAIT/TERRAIN/CONTEXT 16~31가지 조합 |
| **센서 ablation** | Foot → +Shank → +Thigh → +Hand → +Pelvis → 전체 누적 추가 |
| **LOSO** | 피험자 단위 leave-one-out 일반화 검증 |

---

## 실행

```bash
# 1) 환경 초기화
bash scripts/setup.sh

# 2) 단위 테스트 (모듈 import/shape/forward 확인)
PYTHONPATH=. python tests/test_sanity.py

# 3) 빠른 통합 테스트 (에러 없이 돌아가는지 확인, ~5분)
PYTHONPATH=. python -m src.experiments.run_all --fast

# 4) 본실험 (기본 피처 조합, ~8시간)
nohup PYTHONPATH=. python -m src.experiments.run_all \
  > logs/run.log 2>&1 &

# 5) 본실험 (피처 조합 전체 16가지, ~20시간)
nohup PYTHONPATH=. python -m src.experiments.run_all \
  --all_feat_combos > logs/run_all.log 2>&1 &

# 6) 특정 phase만
PYTHONPATH=. python -m src.experiments.run_all --phase 1  # 모델 비교만
PYTHONPATH=. python -m src.experiments.run_all --phase 4  # LOSO만

# 7) 로그 모니터링
tail -f logs/run.log
```

---

## 검증 방식

- **5-fold CV**: Subject-wise StratifiedGroupKFold (데이터 내 교차 검증)
- **LOSO**: Leave-One-Subject-Out (새 피험자 일반화 성능)

---

## 파이럿 실험 결과 (M7_Attr, 5-fold)

| 조건 | 클래스 수 | Acc | Macro F1 |
|------|-----------|-----|----------|
| A 전체 | 6 | 77.7% | 0.753 |
| B 흙길 제외 | 5 | 89.8% | 0.872 |
| C 잔디 제외 | 5 | 87.0% | 0.840 |

> C4(흙길) ↔ C5(잔디) 상호 혼동이 주요 과제
