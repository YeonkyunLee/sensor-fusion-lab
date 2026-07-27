"""잔차 RL(안전한 고전 base + 학습된 보정) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "residual33", ROOT / "scripts" / "33_residual_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_residual_beats_base_and_matches_scratch_at_equal_budget():
    mod = _load()
    (base_cost, scratch_cost, residual_cost,
     base_sse, res_sse, scratch_fall, residual_fall) = mod.main()

    # (i) base+residual 이 '실제 플랜트'에서 base 단독을 확연히 이긴다.
    #     외란이 남긴 정상상태 오차를 보정이 제거하므로 비용이 급감.
    assert residual_cost < 0.25 * base_cost, (
        f"residual({residual_cost}) must clearly beat base({base_cost})")
    assert res_sse < 0.15 * base_sse, (
        f"residual should cancel the steady-state error: "
        f"base_sse={base_sse}, res_sse={res_sse}")

    # (ii) 동일 CEM 예산에서 base+residual 은 from-scratch 이상(이상적으로 더 우수).
    #      base가 대부분을 처리하므로 residual은 작은 보정만 배우면 된다.
    assert residual_cost <= scratch_cost + 1e-6, (
        f"residual({residual_cost}) should be at least as good as "
        f"scratch({scratch_cost}) at equal budget")

    # (iii) 학습 중 안전성: residual은 base가 이미 안정화하므로 거의 넘어지지 않지만
    #       from-scratch는 안정화 이득을 탐색하며 위험 영역을 훨씬 자주 방문한다.
    assert residual_fall < 0.05, f"residual should stay safe while learning: {residual_fall}"
    assert scratch_fall > 3 * residual_fall + 0.03, (
        f"from-scratch should be far less safe during learning: "
        f"scratch_fall={scratch_fall}, residual_fall={residual_fall}")

    # sanity: base는 폴은 세우지만(전도 아님) 큰 정상상태 오차를 남긴다.
    assert base_sse > 0.3, f"base should show a clear steady-state error: {base_sse}"
