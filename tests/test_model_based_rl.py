"""모델기반 RL(동역학 모델 학습 + MPC 계획) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "mbrl38", ROOT / "scripts" / "38_model_based_rl.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mbrl_is_sample_efficient_and_approaches_oracle():
    mod = _load()
    mbrl_at, mf_at, oracle_perf, mbrl_trans_to_solve = mod.main()

    # sanity: 오라클(진짜 모델 MPC)이 상한선으로서 실제로 잘 균형잡는다.
    assert oracle_perf > 0.75, f"oracle MPC should control well: {oracle_perf}"

    # (i) 동일한 '소예산'(COMPARE_BUDGET transitions)에서 MBRL 이 무모델 정책탐색을
    #     확연히 이긴다. 모델을 먼저 배우면 계획은 공짜라 표본효율이 압도적이다.
    assert mbrl_at > 0.75, f"MBRL should reach good control at small budget: {mbrl_at}"
    assert mf_at < 0.45, f"model-free should still be poor at small budget: {mf_at}"
    assert mbrl_at > mf_at + 0.4, (
        f"MBRL({mbrl_at}) must clearly beat model-free({mf_at}) at equal budget")
    assert mbrl_at > 3.0 * mf_at, (
        f"MBRL should be far more sample-efficient: MBRL {mbrl_at} vs MF {mf_at}")

    # (ii) MBRL 은 소량의 전이만으로 오라클에 근접한다(정직한 gap 이내, 그 아래 정체).
    assert mbrl_at >= 0.85 * oracle_perf, (
        f"MBRL({mbrl_at}) should approach the oracle({oracle_perf}) within gap")
    assert mbrl_at <= oracle_perf + 1e-6, (
        f"oracle({oracle_perf}) should stay the upper bound vs MBRL({mbrl_at})")

    # (iii) MBRL 은 아주 적은 전이로 'solved'(오라클의 SOLVE_FRAC) 에 도달한다.
    assert mbrl_trans_to_solve is not None, "MBRL should solve within the budget grid"
    assert mbrl_trans_to_solve <= mod.COMPARE_BUDGET, (
        f"MBRL should solve with few transitions: {mbrl_trans_to_solve}")
