"""도메인 랜덤화 sim-to-real 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "dr30", ROOT / "scripts" / "30_domain_randomization.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dr_transfers_more_robustly_than_nominal_only():
    mod = _load()
    nom_at_nominal, dr_at_nominal, nom_robust, dr_robust = mod.main()

    # (i) 두 정책 모두 명목(sim) 세계에서는 잘 균형잡는다 (과제 자체는 풀린다)
    assert nom_at_nominal >= 0.9, f"nominal-only fails at nominal: {nom_at_nominal}"
    assert dr_at_nominal >= 0.9, f"DR fails at nominal: {dr_at_nominal}"

    # (ii) 이동된/명목밖 격자 평균에서 DR이 명목전용보다 확연히 강건하다
    assert dr_robust > nom_robust + 0.2, (
        f"DR robust {dr_robust} not clearly above nominal-only {nom_robust}")
    assert dr_robust > 1.8 * nom_robust, (
        f"DR robust {dr_robust} should be far above nominal-only {nom_robust}")
    # 명목전용은 이동에 취약, DR은 넓게 견딘다 (정직한 크로스오버)
    assert nom_robust < 0.35, f"nominal-only unexpectedly robust: {nom_robust}"
    assert dr_robust > 0.4, f"DR not robust enough across shifted grid: {dr_robust}"
