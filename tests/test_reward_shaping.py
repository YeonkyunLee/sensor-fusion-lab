"""보상 설계(reward shaping) & reward hacking 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "reward32", ROOT / "scripts" / "32_reward_shaping.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reward_design_determines_learning_and_hacking():
    mod = _load()
    sparse_s, shaped_s, hacked_s, hacked_reward_frac = mod.main()

    # (i) 동일 과제·동일 CEM에서 '잘 설계된' 보상은 참-과제 성공률이
    #     '희소' 보상보다 확연히 높다 → 보상 설계가 학습의 성패를 가른다.
    assert shaped_s >= 0.6, f"shaped should solve the task: {shaped_s}"
    assert shaped_s > sparse_s + 0.2, (
        f"shaped({shaped_s}) must clearly beat sparse({sparse_s})")

    # (ii) '오설계' 보상은 학습 보상은 높지만(회전으로 채움) 참-과제 성공은
    #      바닥이다 → reward hacking. 높은 보상 ≠ 과제 해결.
    assert hacked_reward_frac > 0.4, (
        f"hacked policy should rack up high training reward: {hacked_reward_frac}")
    assert hacked_s < 0.1, f"hacked policy must NOT actually solve the task: {hacked_s}"
    assert shaped_s > hacked_s + 0.4, (
        f"shaped truly solves; hacked only games the reward "
        f"(shaped={shaped_s}, hacked={hacked_s})")


def test_success_metric_rejects_spinning():
    """참-성공 지표는 학습 보상과 무관하며, 빠르게 스쳐 지나가는 회전은 성공이 아니다."""
    import numpy as np

    mod = _load()
    T = mod.T
    # 상단을 매 스텝 통과하지만 고속으로 회전하는 궤적: 각도는 상단이라도 속도 게이트로 탈락.
    ths = np.zeros(T + 1)                       # 항상 상단(angle=0)
    tds = np.full(T + 1, 6.0)                   # 그러나 고속 회전(>TOL_VEL)
    assert mod.true_success(ths, tds) == 0.0

    # 실제로 상단에 저속으로 머물면 성공.
    tds_slow = np.zeros(T + 1)
    assert mod.true_success(ths, tds_slow) == 1.0
