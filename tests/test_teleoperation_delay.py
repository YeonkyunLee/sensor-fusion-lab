"""원격조작 지연·수동성·가상 고정구(exp 50) 테스트. 실행: pytest -q"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "tele50", ROOT / "scripts" / "50_teleoperation_delay.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tissue_model_punctures_and_softens():
    """조직: 누를수록 단단해지다 임계에서 관통하고, 이후 절삭+마찰로 낮아진다."""
    m = _load()
    t = m.Tissue()
    assert t.force(m.X_SURFACE - 0.001) == 0.0          # 접촉 전
    f1 = abs(t.force(m.X_SURFACE + 0.002))
    f2 = abs(t.force(m.X_SURFACE + 0.005))
    assert 0 < f1 < f2
    assert not t.punctured
    f3 = abs(t.force(m.X_SURFACE + 0.02))               # 관통
    assert t.punctured and f3 < m.F_PUNC


def test_direct_force_reflection_degrades_with_delay():
    """P-P 직접 힘반사: 지연이 커지면 정착 구간 진동이 커진다(교과서적 결과)."""
    m = _load()
    small = m.run(arch="pp", delay_ms=5.0)
    large = m.run(arch="pp", delay_ms=100.0)
    assert small["usable"], "작은 지연에서는 안정해야 한다"
    assert not large["usable"], "큰 지연에서 진동이 나타나야 한다"
    assert large["osc_mm"] > 5 * small["osc_mm"]


def test_wave_variables_stay_stable_at_large_delay():
    """파동변수: 200 ms 편도 지연에서도 진동 없이 유지된다."""
    m = _load()
    for d in (20.0, 100.0, 200.0):
        r = m.run(arch="wavepos", delay_ms=d, b_wave=10.0)
        assert not r["diverged"], f"{d} ms 에서 발산"
        assert r["usable"], f"{d} ms 에서 진동 {r['osc_mm']:.2f} mm"


def test_communication_channel_is_passive_for_wave_variables():
    """수동성: 채널이 에너지를 만들지 않는다(들어간 − 나온 ≥ 0)."""
    m = _load()
    for d in (20.0, 100.0):
        r = m.run(arch="wavepos", delay_ms=d, b_wave=10.0)
        e = r["log"]["e_ch"]
        assert e[-1] >= -1e-6, f"{d} ms: 채널 에너지 {e[-1]:.4f} J < 0"
        assert np.min(e) >= -1e-6


def test_force_feedback_improves_transparency_over_unilateral():
    """파동변수는 손이 느끼는 힘을 실제 조직력에 가깝게 만든다(단방향은 0)."""
    m = _load()
    uni = m.run(arch="uni", delay_ms=20.0)
    wav = m.run(arch="wavepos", delay_ms=20.0, b_wave=10.0)
    assert np.allclose(uni["log"]["fm"], 0.0), "단방향인데 힘이 전달됨"
    assert wav["force_err_N"] < uni["force_err_N"] / 3


def test_pp_reflects_the_coupling_spring_not_the_environment():
    """P-P 의 알려진 약점: 손이 느끼는 것은 조직이 아니라 결합 스프링이다."""
    m = _load()
    pp = m.run(arch="pp", delay_ms=20.0)
    wav = m.run(arch="wavepos", delay_ms=20.0, b_wave=10.0)
    assert pp["force_err_N"] > 3 * wav["force_err_N"]
    assert pp["force_err_N"] > pp["max_force"], "결합 스프링 힘이 조직력보다 크게 느껴져야 한다"


def test_position_correction_reduces_wave_variable_drift():
    """파동변수의 위치 표류를 위치 보정이 줄인다."""
    m = _load()
    plain = m.run(arch="wave", delay_ms=20.0, b_wave=10.0)
    fixed = m.run(arch="wavepos", delay_ms=20.0, b_wave=10.0)
    assert fixed["pos_err_mm"] < plain["pos_err_mm"]


def test_virtual_fixture_must_be_rendered_locally():
    """가상 고정구: 로컬 렌더링은 강성을 올릴 수 있지만, 지연 경로로 렌더링하면 무너진다."""
    m = _load()
    no_wall = m.run(arch="uni", delay_ms=50.0, vf_stiffness=0.0)
    local = m.run(arch="uni", delay_ms=50.0, vf_stiffness=12000.0, vf_mode="local")
    remote = m.run(arch="uni", delay_ms=50.0, vf_stiffness=12000.0, vf_mode="remote")

    assert local["wall_pen_mm"] < no_wall["wall_pen_mm"] / 10, "로컬 벽이 침범을 못 막음"
    assert not local["diverged"]
    # 같은 강성을 지연 경로로 렌더링하면 발산하거나 훨씬 크게 침범한다
    assert remote["diverged"] or remote["wall_pen_mm"] > 5 * local["wall_pen_mm"]


def test_main_runs_and_reports():
    """main() 전체 실행 — 표·그림 생성 경로가 살아 있는지."""
    m = _load()
    res = m.main()
    assert set(res["table"]) == {"uni", "pp", "wave", "wavepos"}
    assert len(res["vf_rows"]) >= 5
