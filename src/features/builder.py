# -*- coding: utf-8 -*-
"""
src/features/builder.py — 피처 타입별 인덱스 맵 & 조합 생성

피처 구성 (390차원):
  Pelvis(30) + Hand(38) + Thigh(36) + Shank(40) + Foot(88)
  + Terrain(98) + Context(60) = 390

피처 타입:
  TIME    : 시간 도메인 통계
  FREQ    : 주파수 도메인
  GAIT    : 보행 특화
  TERRAIN : 지형 충격
  CONTEXT : bout 컨텍스트
"""
from __future__ import annotations
import numpy as np
from itertools import combinations

SENSOR_GROUPS: dict[str, tuple[int, int]] = {
    "Pelvis":  (0,   30),
    "Hand":    (30,  68),
    "Thigh":   (68,  104),
    "Shank":   (104, 144),
    "Foot":    (144, 232),
    "Terrain": (232, 330),
    "Context": (330, 390),
}

SENSOR_ADD_ORDER  = ["Foot", "Shank", "Thigh", "Hand", "Pelvis", "Terrain"]
FEAT_TYPES        = ["TIME", "FREQ", "GAIT", "TERRAIN", "CONTEXT"]
FEAT_TYPES_NO_CTX = ["TIME", "FREQ", "GAIT", "TERRAIN"]


def _build_type_map() -> dict[str, list[int]]:
    m: dict[str, list[int]] = {t: [] for t in FEAT_TYPES}
    # Pelvis 0~29
    m["TIME"] += list(range(0, 14)); m["FREQ"] += list(range(14, 21))
    m["GAIT"] += list(range(21, 26)); m["TIME"] += list(range(26, 30))
    # Hand 30~67
    m["TIME"] += list(range(30, 37)); m["FREQ"] += list(range(37, 44))
    m["GAIT"] += list(range(44, 49)); m["TIME"] += list(range(49, 56))
    m["FREQ"] += list(range(56, 63)); m["GAIT"] += list(range(63, 68))
    # Thigh 68~103
    m["TIME"] += list(range(68, 77)); m["FREQ"] += list(range(77, 82))
    m["GAIT"] += list(range(82, 86)); m["TIME"] += list(range(86, 95))
    m["FREQ"] += list(range(95, 100)); m["GAIT"] += list(range(100, 104))
    # Shank 104~143
    m["TIME"] += list(range(104, 116)); m["GAIT"] += list(range(116, 122))
    m["TIME"] += list(range(122, 134)); m["GAIT"] += list(range(134, 144))
    # Foot 144~231
    m["TIME"] += list(range(144, 168)); m["FREQ"] += list(range(168, 175))
    m["GAIT"] += list(range(175, 186)); m["TIME"] += list(range(186, 210))
    m["FREQ"] += list(range(210, 217)); m["GAIT"] += list(range(217, 232))
    # Terrain 232~329
    m["TERRAIN"] += list(range(232, 248))   # C4/C5 특화
    m["GAIT"]    += list(range(248, 256))   # C6 규칙성
    m["TERRAIN"] += list(range(256, 264))   # 공통
    m["GAIT"]    += list(range(264, 278))   # autocorr/cross_corr
    m["TIME"]    += list(range(278, 282))   # skewness
    m["FREQ"]    += list(range(282, 306))   # fine_band/inst_freq
    m["GAIT"]    += list(range(306, 309))   # gait_cycle_var
    m["TERRAIN"] += list(range(309, 330))   # G-group
    # Context 330~389
    m["CONTEXT"] += list(range(330, 390))
    return m


_TYPE_MAP = _build_type_map()


def get_feat_mask(feat_types: list[str], n_features: int = 390) -> np.ndarray:
    """피처 타입 조합 → 인덱스 배열"""
    idx = [i for t in feat_types for i in _TYPE_MAP.get(t, []) if i < n_features]
    return np.array(sorted(set(idx)), dtype=np.int64)


def get_sensor_mask(sensors: list[str], n_features: int = 390) -> np.ndarray:
    """센서 그룹 조합 → 인덱스 배열"""
    idx = []
    for s in sensors:
        if s in SENSOR_GROUPS:
            s0, s1 = SENSOR_GROUPS[s]
            idx.extend(range(s0, min(s1, n_features)))
    return np.array(sorted(set(idx)), dtype=np.int64)


def all_feat_type_combos(include_context: bool = False) -> list[list[str]]:
    """2^4=16 or 2^5=32 가지 조합"""
    types = FEAT_TYPES if include_context else FEAT_TYPES_NO_CTX
    return [list(c) for r in range(1, len(types)+1) for c in combinations(types, r)]


def sensor_cumulative_combos() -> list[list[str]]:
    """Foot → +Shank → ... → 전체 누적"""
    result, acc = [], []
    for s in SENSOR_ADD_ORDER:
        acc = acc + [s]; result.append(acc.copy())
    return result


def combo_name(combo: list[str]) -> str:
    abbr = {"TIME":"T","FREQ":"F","GAIT":"G","TERRAIN":"Tr","CONTEXT":"C",
            "Pelvis":"Pv","Hand":"Hd","Thigh":"Th","Shank":"Sk",
            "Foot":"Ft","Terrain":"Trn","Context":"Ctx"}
    return "+".join(abbr.get(c, c[:3]) for c in combo) if combo else "none"
