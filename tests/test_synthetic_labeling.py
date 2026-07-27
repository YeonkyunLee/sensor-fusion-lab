"""합성 데이터 & 자동 라벨링 sim-to-real 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "syn34", ROOT / "scripts" / "34_synthetic_labeling.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_domain_randomization_makes_synthetic_labels_transfer():
    mod = _load()
    scarce_rmse, noDR_rmse, DR_rmse = mod.main()

    # 모든 지표는 유한한 양의 픽셀 오차
    for v in (scarce_rmse, noDR_rmse, DR_rmse):
        assert v > 0 and v < 50, f"RMSE out of sane range: {v}"

    # (i) 합성+DR 이 합성-DR없음(명목 외형 과적합)보다 확연히 낮다
    assert DR_rmse < 0.6 * noDR_rmse, (
        f"DR {DR_rmse:.3f} not clearly below no-DR {noDR_rmse:.3f}")
    assert noDR_rmse - DR_rmse > 2.0, (
        f"DR gain over no-DR too small: {noDR_rmse - DR_rmse:.3f}px")

    # (ii) 합성+DR 이 희소 현실(40장 손라벨) 대비 최소한 경쟁력 있다(이상적으로 더 낫다)
    assert DR_rmse <= scarce_rmse + 0.3, (
        f"DR {DR_rmse:.3f} not competitive with scarce-real {scarce_rmse:.3f}")

    # 합성-DR없음은 현실에서 실제로 무너져야 한다(단조 외형 과적합의 정직한 증거)
    assert noDR_rmse > 3.0, f"no-DR unexpectedly robust: {noDR_rmse:.3f}"
