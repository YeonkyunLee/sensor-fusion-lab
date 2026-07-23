"""VIO · 안전 자율성 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vio_beats_imu_only():
    mod = _load("08_vio.py")
    imu_rmse, vio_rmse = mod.main()
    assert vio_rmse < imu_rmse * 0.6   # 시각 융합이 드리프트를 확실히 줄임


def test_uncertainty_aware_prevents_violations():
    mod = _load("09_safe_autonomy.py")
    naive_viol, aware_viol = mod.main()
    assert aware_viol < 0.05           # 불확실도-인지: 침범 거의 0
    assert naive_viol > aware_viol     # naive보다 안전
