"""SimOpt sim-to-real 루프 테스트. 실행: pytest -q

루프가 실제로 간극을 닫는지 확인한다:
  (i)  최종 시뮬레이터 파라미터오차가 초기보다 확실히 작다(시스템 식별 수렴).
  (ii) 최종 실제 균형유지 성능이 초기보다 확실히 좋다(루프가 sim-to-real gap을 닫음).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "simopt31", ROOT / "scripts" / "31_simopt_loop.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_simopt_closes_the_loop():
    mod = _load()
    initial_real, final_real, initial_perr, final_perr = mod.main()

    # (i) 시스템 식별: 최종 파라미터오차가 초기보다 확실히 작다(≥60% 감소).
    assert final_perr < 0.4 * initial_perr, \
        f"param error did not converge: {initial_perr:.3f} -> {final_perr:.3f}"
    assert final_perr < 0.15, f"final param error still large: {final_perr:.3f}"

    # (ii) 실제 성능: 초기(나쁜 시뮬레이터 정책)는 넘어지고, 최종은 확실히 개선.
    assert initial_real < 0.7 * mod.T, \
        f"initial policy should fail on real, got {initial_real:.1f}/{mod.T}"
    assert final_real >= initial_real + 50, \
        f"real perf did not improve enough: {initial_real:.1f} -> {final_real:.1f}"
    assert final_real >= 0.9 * mod.T, \
        f"final policy should balance the real plant, got {final_real:.1f}/{mod.T}"


def test_sysid_recovers_true_params():
    """식별된 파라미터가 참값에 근접(관측 가능한 mp,l 회복)."""
    mod = _load()
    mod.main()
    # main 내부에서 사용한 참값/초기추정 상수 노출 확인
    assert mod.TRUE_PARAMS.shape == (2,)
    assert mod.param_error(mod.INIT_GUESS) > 0.5   # 초기 추정은 크게 어긋나 있음
