"""증분 스무딩(iSAM 계열) 테스트. 실행: pytest -q

deterministic(고정 시드) 이므로 main() 반환값에 직접 단언한다.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _run():
    spec = importlib.util.spec_from_file_location(
        "s35", ROOT / "scripts" / "35_incremental_smoothing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()   # (batch_rmse, incremental_rmse, batch_cost, incremental_cost)


def test_incremental_matches_batch_accuracy():
    rb, ri, cb, ci = _run()
    # 증분해가 배치해에 근접: 20% 이내 또는 작은 절대 격차(<0.15 m)
    assert ri <= rb * 1.2 or (ri - rb) < 0.15


def test_incremental_much_cheaper_than_batch():
    rb, ri, cb, ci = _run()
    # 증분해의 누적 계산량이 배치 재풀이보다 뚜렷이 작아야 함(절반 미만)
    assert ci < cb * 0.5


def test_headline_metrics_sane():
    rb, ri, cb, ci = _run()
    assert rb > 0 and ri > 0
    assert ri < 1.0          # 배치급 정확도(서브미터)
    assert cb > 0 and ci > 0
