"""재활 외골격 보행 위상(gait-phase) 추정 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gait_detection_and_zupt():
    mod = _load("27_gait_estimation.py")
    acc, ev_err, zupt_err, naive_err = mod.main()
    # 입각/유각 분류 정확도 > 90%
    assert acc > 0.90
    # 사건(heel-strike/toe-off) 타이밍 오차가 작음(< 60 ms)
    assert ev_err < 60.0
    # ZUPT stride 오차가 순진 이중적분보다 확실히(>=5배) 작음
    assert zupt_err < naive_err / 5.0
    # ZUPT stride 오차 자체도 작음(< 10 cm on 70 cm stride)
    assert zupt_err < 0.10


def test_events_all_detected():
    mod = _load("27_gait_estimation.py")
    out = mod.synth(seed=0)
    stance_true, hs_true, to_true = out[4], out[5], out[6]
    stance_det = mod.detect_stance(out[1], out[3])
    hs_det, to_det = mod.transitions(stance_det)
    # 검출된 heel-strike/toe-off 개수가 참값과 일치
    assert len(hs_det) == len(hs_true)
    assert len(to_det) == len(to_true)
