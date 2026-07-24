"""MPC 궤적 추종 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("mpc24", ROOT / "scripts" / "24_mpc_tracking.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mpc_tracks_tighter_and_respects_limits():
    mod = _load()
    mpc_rmse, pp_rmse = mod.main()

    # 1) MPC가 기준궤적을 조밀하게 추종(절대 임계)
    assert mpc_rmse < 0.05, f"MPC RMSE too high: {mpc_rmse}"

    # 2) MPC가 pure-pursuit 기준선보다 확실히 우수(최소한 동등, 실제로는 크게 우수)
    assert mpc_rmse <= pp_rmse, f"MPC({mpc_rmse}) should beat pursuit({pp_rmse})"
    assert mpc_rmse < 0.7 * pp_rmse, "MPC should be clearly tighter than pure-pursuit"


def test_mpc_controls_within_actuator_limits():
    mod = _load()
    Xref, Uref = mod.make_reference()
    _, mpc_u = mod.run_mpc(Xref, Uref)
    v = np.abs(mpc_u[:, 0])
    w = np.abs(mpc_u[:, 1])
    assert v.max() <= mod.V_MAX + 1e-9, f"|v| exceeds limit: {v.max()}"
    assert w.max() <= mod.W_MAX + 1e-9, f"|w| exceeds limit: {w.max()}"
    # 가속 한계도 대체로 만족(적용 제어의 스텝간 변화)
    dv = np.abs(np.diff(v)) / mod.DT
    dw = np.abs(np.diff(w)) / mod.DT
    assert dv.max() <= mod.A_V + 1e-6
    assert dw.max() <= mod.A_W + 1e-6
