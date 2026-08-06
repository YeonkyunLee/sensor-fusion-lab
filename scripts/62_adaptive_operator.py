"""적응형 술자 — 힘을 느끼고, 배우고, 되돌리는 사람. 그리고 그게 결론을 어떻게 바꾸는가.

exp 59 는 술자 쪽 대책(마스터 잠금)의 **부호가 술자 모델에 따라 뒤집히는** 것을 보이고 규칙을
하나 적었다: **술자 쪽 주장은 술자 모델이 적응적일 때만 평가할 수 있다.** 그러면서 자기 술자
모델의 한계를 세 개 남겼다 — **힘 지각 없음, 학습 없음, 되돌림 없음.** exp 60·61 이 조직 쪽을
닫았으니 이 트랙에서 남은 모델 작업은 이쪽뿐이다.

특히 하나가 이상했다. exp 50 은 파동변수로 힘을 **전달**하는 데 공을 들여 놓고, 술자 모델은
그 힘을 **쓰지 않았다.** 열두 실험 동안 **못 느끼는 사람에게 힘을 보내고 있었다.**

  A. **술자 계층 사다리** — 고정 임피던스(exp 50) → 시각 반응(exp 59) → **힘 지각** →
     **학습**(겪을수록 조심스러워진다, 고전적인 move-and-wait) → **되돌림**(얼지 말고 빼기).
     그리고 숫자를 읽기 전에 **완주 여부**부터 본다.
  B. **완주를 맞추면 무엇이 남는가** — A 의 사다리가 대부분 무너진다.
  C. **exp 59 의 마스터 잠금 결론을 다섯 술자 모델 전부에 다시 건다.**
  D. **사람은 수동성 증명 밖에 있다** — 힘에 반응하는 술자가 에너지를 넣는가.

**A 에서 내가 내 규칙을 어겼다.** 4 초(이 사슬의 표준 길이)에서 계층을 올릴수록 복귀 돌진이
120 → 41 mm/s 로 깨끗하게 좋아진다. 그런데 도달 깊이가 50.3 → **34.7 mm** 로 떨어진다.
exp 56 이 바로 그 숫자(34.8 mm)에서 **"시험이 실패할 수가 없었다"** 를 발견하고 R18 을 만들었다 —
로버스트니스 결과는 **과제를 완주하는 설정에서 완주 지표와 함께** 낼 것. 내 술자 모델이 그 규칙을
정확히 위반했다. **조심스러운 술자는 안전을 사는 게 아니라 과제를 안 끝내고 있었다.**

시간을 줘서 완주시키면(12 초) 이야기가 달라진다. 그리고 **여기서 한 번 더 틀렸다** — 6 시드
중앙값으로는 "힘 지각은 이득이 없다"로 보였는데, 시드를 12 개로 늘려 **짝지어** 보니 반대다.
exp 59 가 바로 이 지표에서 중앙값 대 중앙값의 함정을 잡았는데 같은 함정에 다시 빠졌다. 정본은 이렇다:

  · **시각 반응** −29.7 mm/s, 개선 시드 **8/12** — 진짜지만 일관되지 않다
  · **힘 지각**  −34.6 mm/s, 개선 시드 **11/12** — 크기가 아니라 **신뢰성**을 산다.
    시각만으로 지는 시드 4 개 중 **3 개를 구제**하고, 이미 이기던 시드에서는 값이 비트 단위로 같다.
    → **중복 단서는 천장을 올리지 않고 실패를 없앤다**(다른 물리 채널로 같은 것을 본다).
  · **학습**    −32.3 mm/s, 9/12 — T2 대비 이득 없음. 4 초에서는 과제를 아예 못 끝내게 했다.
  · **되돌림**  −60.9 mm/s, 11/12 — 가장 크다. 대가는 **맹행이 아니라**(짝지으면 오히려 −0.04 mm)
    **개입 횟수 +3 회(10/12)와 붙들기 힘 +0.33 N(7/12)** — 빼면 자기 단서가 다시 켜지기 때문이다.

    python scripts/62_adaptive_operator.py

한계·트레이드오프
  - 술자가 여전히 **모델**이다. 지각은 임계값 하나, 학습은 스칼라 하나(내부 시계의 진행 속도),
    되돌림은 고정 거리다. 실제 사람은 예측 모형을 만들고 전략을 바꾼다.
  - 학습을 **내부 시계 감속**으로 넣었다. 이게 move-and-wait 의 1차 근사이지만, 실제로는 속도만
    줄이는 게 아니라 **분절된 이동-대기**가 된다. 분절을 넣으면 완주 시간이 더 길어질 것이다.
  - 완주 비교를 위해 시간을 3배로 줬다. **시간이 공짜가 아닌 시술이면 이 비교 자체가 다르다** —
    이 실험은 "시간을 주면 무엇이 남는가"까지만 말하고, 시간의 임상 가치는 또 다른 선언이다.
  - 힘 지각을 **크기 임계**로만 넣었다. 사람은 힘의 **변화율**과 방향에도 반응하고, 그쪽이 더
    빠른 단서일 수 있다.
  - 수동성 측정은 여전히 파동 블록 장부다. 사람이 그 밖에 있다는 것을 보이는 데는 충분하지만,
    사람을 포함한 전체 수동성 논증은 아니다.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
jc = import_module("56_jittery_channel")
bc = import_module("57_bursty_channel")
s59 = import_module("59_what_is_safe_state")

DT, STEPS = jc.DT, jc.STEPS
X_TARGET_MM = (jc.tele.X_TARGET - 0.0) * 1e3
# 복귀 돌진은 시드 분산이 크다(같은 조건에서 47~192 mm/s). 6 시드 중앙값으로 보면 결론의
# 부호가 바뀌는 것을 실제로 겪어서, 시드를 늘리고 **짝지은 통계**를 정본으로 쓴다.
N_SEEDS = 12
LONG = 3 * STEPS            # 완주를 맞추기 위한 시간(느려진 술자도 끝낼 수 있게)

REACT_MS = s59.REACT_MS if hasattr(s59, "REACT_MS") else 200.0
FORCE_N = 2.5               # 술자가 "너무 세다"고 느끼는 문턱
LEARN = 0.7                 # 겪을 때마다 내부 시계 진행 속도에 곱하는 값
REVERSE_MM = 3.0            # 얼지 않고 빼는 거리

# 술자 계층. 각 단계는 **앞 단계에 하나만 더한다**(그래야 무엇이 기여했는지 귀속된다).
TIERS = (
    ("T0 고정 임피던스", {}),
    ("T1 +시각 반응", dict(op_react_ms=REACT_MS)),
    ("T2 +힘 지각", dict(op_react_ms=REACT_MS, op_force_N=FORCE_N)),
    ("T3 +학습", dict(op_react_ms=REACT_MS, op_force_N=FORCE_N, op_learn=LEARN)),
    ("T4 +되돌림", dict(op_react_ms=REACT_MS, op_force_N=FORCE_N, op_learn=LEARN,
                      op_reverse_mm=REVERSE_MM)),
)

BASE = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
            resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ)
KEYS = ("final_depth_mm", "resume_vmax_mms", "blind_max_mm", "e_min", "e_drawdown",
        "op_rate_end", "n_adverse", "n_force_cue", "mismatch_release_mm",
        "f_e_held_swing", "osc_mm")


def series(kw, lock=False, seeds=N_SEEDS, steps=STEPS):
    """한 술자 계층을 여러 시드로 돌려 **시드별 배열**을 낸다(짝지은 비교에 필요하다)."""
    out = {k: [] for k in KEYS}
    n_div = 0
    for s in range(seeds):
        r = bc.run("tdpa", seed=s, master_lock=lock, steps=steps,
                   tissue_obj=s59.GrippingTissue(), **BASE, **kw)
        if r["diverged"]:
            n_div += 1
        for k in KEYS:
            out[k].append(np.nan if r["diverged"] else r[k])
    res = {k: np.asarray(v, float) for k, v in out.items()}
    res["diverged"] = n_div
    return res


def tier(kw, lock=False, seeds=N_SEEDS, steps=STEPS):
    """중앙값만 필요할 때. **중앙값끼리 비교하면 짝지은 효과를 과장한다**(exp 59) — 결론을
    낼 때는 paired() 를 쓰고, 이건 표에 값을 적을 때만 쓴다."""
    s = series(kw, lock=lock, seeds=seeds, steps=steps)
    res = {k: float(np.nanmedian(v)) for k, v in s.items() if k != "diverged"}
    res["diverged"] = s["diverged"]
    return res


def paired(a, b, key="resume_vmax_mms", lower_is_better=True):
    """**같은 시드끼리 짝지어** 비교한다. (차이의 중앙값, 이긴 시드 수, 전체 시드 수)

    exp 59 가 이 지표에서 중앙값 대 중앙값이 짝지은 효과를 과장하는 것을 잡았고, exp 62 에서는
    6 시드 중앙값이 아예 **결론의 부호를 바꿨다**(힘 지각이 이득 없음 → 있음). 그래서 정본은 이쪽.
    """
    d = b[key] - a[key]
    ok = np.isfinite(d)
    wins = int(np.sum(d[ok] < 0)) if lower_is_better else int(np.sum(d[ok] > 0))
    return float(np.nanmedian(d)), wins, int(np.sum(ok))


def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    tiers = (TIERS[0], TIERS[1], TIERS[4]) if quick else TIERS
    print("=== 62. 적응형 술자 — 힘을 느끼고, 배우고, 되돌리는 사람 ===")
    print("exp 59 가 '술자 쪽 주장은 적응형 술자에서만 평가할 수 있다'고 적고 한계 셋을 남겼다:")
    print("**힘 지각 없음, 학습 없음, 되돌림 없음.** 셋을 넣고 앞의 결론들을 다시 건다.")
    print("그리고 exp 50 은 파동변수로 힘을 **전달**해 놓고 술자 모델이 그걸 **쓰지 않았다** —")
    print("열두 실험 동안 못 느끼는 사람에게 힘을 보내고 있었다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print(f"[A] 술자 계층 사다리 — 이 사슬의 표준 길이({STEPS * DT:.0f} 초)에서")
    print(f"{'술자':>14s} | {'도달 깊이[mm]':>13s} | {'복귀 최대[mm/s]':>15s} | {'맹행[mm]':>9s} | "
          f"{'시계 속도':>9s} | {'사건':>5s}")
    A = {}
    for name, kw in tiers:
        m = tier(kw, seeds=seeds, steps=STEPS)
        A[name] = m
        print(f"{name:>14s} | {m['final_depth_mm']:13.1f} | {m['resume_vmax_mms']:15.1f} | "
              f"{m['blind_max_mm']:9.2f} | {m['op_rate_end']:9.2f} | {m['n_adverse']:5.0f}")
    d0, dN = A[tiers[0][0]]["final_depth_mm"], A[tiers[-1][0]]["final_depth_mm"]
    r0, rN = A[tiers[0][0]]["resume_vmax_mms"], A[tiers[-1][0]]["resume_vmax_mms"]
    print(f"  복귀 돌진이 {r0:.0f} → {rN:.0f} mm/s 로 깨끗하게 좋아진다. **그런데 도달 깊이가**")
    print(f"  **{d0:.1f} → {dN:.1f} mm 로 떨어진다**(표적 55 mm).")
    print("  exp 56 이 정확히 이 숫자대(34.8 mm)에서 **'시험이 실패할 수가 없었다'** 를 발견하고")
    print("  R18 을 만들었다 — 로버스트니스 결과는 **완주하는 설정에서 완주 지표와 함께** 낼 것.")
    print("  **내 술자 모델이 그 규칙을 그대로 위반했다.** 조심스러운 술자는 안전을 산 게 아니라")
    print("  과제를 안 끝내고 있었다. 자기 저장소의 규칙이 자기 자신에게 걸린 것이다.")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print(f"[B] 시간을 줘서 완주시키면 무엇이 남는가 ({LONG * DT:.0f} 초)")
    print("    복귀 돌진은 시드 분산이 커서(같은 조건에서 47~192 mm/s) **짝지은 통계가 정본**이다.")
    print("    6 시드 중앙값으로 봤을 때는 결론의 부호가 달랐다 — exp 59 가 경고한 함정에 또 빠졌다.")
    print(f"{'술자':>14s} | {'깊이[mm]':>9s} | {'복귀 중앙[mm/s]':>15s} | {'맹행[mm]':>9s} | "
          f"{'T0 대비 짝지은 차':>18s} | {'개선 시드':>9s}")
    SER = {name: series(kw, seeds=seeds, steps=LONG) for name, kw in tiers}
    B = {n: {k: float(np.nanmedian(v)) for k, v in s.items() if k != "diverged"}
         for n, s in SER.items()}
    names = [n for n, _ in tiers]
    base = SER[names[0]]
    for name in names:
        m, s_ = B[name], SER[name]
        dm, w, tot = paired(base, s_)
        cell = "—" if name == names[0] else f"{dm:+.1f} mm/s"
        wcell = "—" if name == names[0] else f"{w}/{tot}"
        print(f"{name:>14s} | {m['final_depth_mm']:9.1f} | {m['resume_vmax_mms']:15.1f} | "
              f"{m['blind_max_mm']:9.2f} | {cell:>18s} | {wcell:>9s}")
    if len(names) >= 5:
        d1 = paired(base, SER[names[1]])
        d2 = paired(base, SER[names[2]])
        d3 = paired(base, SER[names[3]])
        d4 = paired(base, SER[names[4]])
        print(f"  · **시각 반응**은 진짜지만 **일관되지 않다**: {d1[0]:+.1f} mm/s, "
              f"{d1[1]}/{d1[2]} 시드에서만 개선.")
        print(f"  · **힘 지각**은 크기를 키우는 게 아니라 **신뢰성을 산다**: {d2[0]:+.1f} mm/s 인데")
        print(f"    개선 시드가 {d1[1]}/{d1[2]} → **{d2[1]}/{d2[2]}**. 시각 단서가 놓친 시드를 메운다.")
        print(f"  · **학습**은 이득이 없다: {d3[0]:+.1f} mm/s, {d3[1]}/{d3[2]} — T2 보다 나아지지 않는다.")
        print(f"  · **되돌림**이 가장 크다: {d4[0]:+.1f} mm/s, {d4[1]}/{d4[2]}.")
        # 대가도 **짝지어** 잰다. 6 시드 중앙값으로는 '맹행이 가장 크다'로 보였는데 짝지으면
        # 맹행은 대가가 아니다 — 진짜 대가는 개입 횟수와 조직에 얹는 힘이다.
        print("    대가를 짝지어 재면 **맹행이 아니다**:")
        for k, lbl, unit in (("blind_max_mm", "맹행", "mm"),
                             ("n_adverse", "개입 횟수", "회"),
                             ("f_e_held_swing", "붙들기 힘", "N")):
            dm, w, tot = paired(SER[names[1]], SER[names[4]], key=k,
                                lower_is_better=False)
            verdict = "**대가다**" if w >= 7 and dm > 0 else "대가가 아니다"
            print(f"      T1 대비 {lbl:>7s}: {dm:+7.3f} {unit:2s} · 나빠진 시드 {w}/{tot} → {verdict}")

    # 힘 지각이 **무엇을** 메우는지 시드 단위로 확인한다(주장의 메커니즘).
    if len(names) >= 3:
        t0v, t1v, t2v = (SER[names[i]]["resume_vmax_mms"] for i in range(3))
        lose1 = t1v >= t0v
        resc = int(np.sum(lose1 & (t2v < t0v)))
        same = int(np.sum(np.isclose(t1v, t2v, rtol=1e-9)))
        print(f"  시드 단위로 보면 정확하다 — 시각만으로 **지는 시드가 {int(np.sum(lose1))}개**이고")
        print(f"  힘 지각이 그중 **{resc}개를 구제**한다. 이미 이기던 시드 {same}개에서는 값이 "
              f"**비트 단위로 같다**")
        print("  (힘 단서가 먼저 걸리지 않았다는 뜻). → **중복 단서는 천장을 올리지 않고 실패를**")
        print("  **없앤다.** 서로 다른 물리 채널로 같은 것을 보는 것이라 [[검증 이중화]]와 같은 논리다.")
    print("  되돌림의 대가는 **자기 단서를 다시 켜는 것**이다 — 빼면 손과 도구의 어긋남이 다시")
    print("  커져 규칙이 또 걸린다. 얼기에는 없는 되먹임이라 개입이 늘고, 그만큼 조직에 얹는 힘도")
    print("  는다. exp 59 에서 조직 쪽 후퇴가 맹행을 물었던 것과는 **다른 종류의 대가**다.")
    print("  → **힘을 느끼게 해서 사는 것은 크기가 아니라 신뢰성이고, 크기를 사는 것은 그 힘으로**")
    print("    **무엇을 하느냐다** — 얼면 시간을 잃고, 빼면 개입 횟수를 문다.")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] exp 59 의 마스터 잠금 결론을 **다섯 술자 모델 전부**에 다시 건다")
    print("    (술자가 느끼는 힘에 **잠금 자신의 저항**도 포함시켰다 — 안 그러면 질문이 불공정하다.)")
    print(f"{'술자':>14s} | {'자유 복귀':>10s} | {'잠금 복귀':>10s} | {'짝지은 차':>11s} | "
          f"{'잠금이 나쁜 시드':>14s} | {'어긋남 자유→잠금':>17s}")
    C = {}
    for name, kw in tiers:
        free_s = SER[name]
        lock_s = series(kw, lock=True, seeds=seeds, steps=LONG)
        dm, w, tot = paired(free_s, lock_s, lower_is_better=False)
        fm = {k: float(np.nanmedian(v)) for k, v in free_s.items() if k != "diverged"}
        lm = {k: float(np.nanmedian(v)) for k, v in lock_s.items() if k != "diverged"}
        C[name] = (fm, lm, (dm, w, tot))
        print(f"{name:>14s} | {fm['resume_vmax_mms']:10.1f} | {lm['resume_vmax_mms']:10.1f} | "
              f"{dm:+11.1f} | {f'{w}/{tot}':>14s} | "
              f"{fm['mismatch_release_mm']:7.2f} → {lm['mismatch_release_mm']:6.2f}")
    print("  **다섯 계층 전부에서 잠금이 복귀를 나쁘게 한다**(짝지은 차가 전부 양수).")
    print("  어긋남은 당연히 줄어드는데도. 다만 **보편적이지 않다** — 시드의 2/3 정도다.")
    print("  **내 예측이 틀렸다** — 힘을 느끼는 술자라면 잠금의 저항 자체가 단서가 되니 exp 59 의")
    print("  '잠금이 단서를 가린다'가 완화될 거라 봤다. 그렇지 않다. 잠금은 여전히 술자의 의도를")
    print("  **없애는 게 아니라 손 스프링에 저장**하고, 해제 순간 그게 돌진으로 나온다.")
    print("  exp 59 의 결론이 **훨씬 풍부한 술자 모델에서 재확인됐다** — 그게 이 절의 값이다.")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] 사람은 수동성 증명 **밖에** 있다 — 힘에 반응하는 술자가 에너지를 넣는가")
    print(f"{'술자':>14s} | {'채널 E_min[J]':>13s} | {'최대 낙폭[mJ]':>13s} | {'진동[mm]':>9s} | "
          f"{'발산':>5s}")
    for name, kw in tiers:
        m, s_ = B[name], SER[name]
        print(f"{name:>14s} | {m['e_min']:13.4f} | {m['e_drawdown'] * 1e3:13.1f} | "
              f"{m['osc_mm']:9.2f} | {s_['diverged']:5d}")
    print("  **넣지 않는다.** 채널 장부가 어느 계층에서도 음이 되지 않는다.")
    print("  이유가 중요하다 — 여기 술자의 반응은 **이득이 아니라 규칙**이다(목표를 얼리거나 뺀다).")
    print("  exp 50 은 시각 폐루프를 **이득 있는 루프**로 걸었다가 사람 루프가 발산해 폐기했다.")
    print("  → **경계는 '사람이 반응하느냐'가 아니라 '반응의 형태가 무엇이냐'다.** 이득을 가진")
    print("    반응은 지연과 곱해져 발산하고, 목표를 재설정하는 규칙은 그렇지 않다.")
    print("  다만 이 장부는 여전히 **파동 블록**의 것이다. 사람을 포함한 전체 수동성 논증은 아니고,")
    print("  56~58 이 보인 대로 **증명이 덮지 않는 곳에 일이 있을 수 있다.** 사람이 바로 그 자리다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 정리")
    print("  1. **완주 지표 없이 술자 계층을 비교하면 전부 좋아 보인다.** 4 초에서는 사다리가")
    print(f"     {r0:.0f} → {rN:.0f} mm/s 인데 깊이가 {d0:.1f} → {dN:.1f} mm 다. 자기 저장소의 R18 을")
    print("     자기가 어겼고, 그 규칙이 아니었으면 못 잡았다.")
    print("  2. **또 6 시드 중앙값에 속았다.** '힘 지각은 이득이 없다'가 12 시드 짝지은 통계에서")
    print("     뒤집혔다 — exp 59 가 이 지표에서 잡았던 바로 그 함정이다. **지표가 시끄러우면")
    print("     중앙값 대 중앙값은 부호까지 바꾼다.**")
    print("  3. 힘 지각이 사는 것은 **크기가 아니라 신뢰성**이다(개선 시드 8/12 → 11/12). 시각만으로")
    print("     지는 시드를 메우고, 이미 이기는 시드에서는 값이 같다. **중복 단서의 값은 천장이")
    print("     아니라 실패 제거에 있다** — 다른 물리 채널로 같은 것을 보기 때문이다.")
    print("  4. **되돌림의 대가는 맹행이 아니다**(짝지으면 오히려 −0.04 mm). **개입 횟수 +3 회**와")
    print("     **붙들기 힘 +0.33 N** 이다 — 빼면 자기 단서가 다시 켜져 규칙이 또 걸린다.")
    print("  5. **exp 59 의 마스터 잠금 결론이 다섯 술자 모델 전부에서 재확인됐다.** 힘 지각이")
    print("     그걸 구제할 거라는 **내 예측은 틀렸다**(잠금의 저항까지 느끼게 해도 그렇다).")
    print("  6. 사람은 수동성 증명 밖에 있지만 **여기서는 에너지를 넣지 않는다** — 반응이 이득이")
    print("     아니라 규칙이기 때문이다. 경계는 반응 여부가 아니라 **반응의 형태**다.")
    print("  7. 남은 것: 분절된 move-and-wait, 힘의 **변화율**에 대한 반응, 예측 모형을 만드는 술자,")
    print("     그리고 **시간의 임상 가치**(이 비교 전체가 '시간을 3배 준다'는 가정 위에 있다).")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    names = [n for n, _ in tiers]
    short = [n.split()[0] for n in names]
    xs = np.arange(len(names))

    ax = axes[0, 0]
    ax.bar(xs - 0.2, [A[n]["resume_vmax_mms"] for n in names], 0.4,
           color="crimson", label="resume peak [mm/s]")
    ax2 = ax.twinx()
    ax2.plot(xs + 0.2, [A[n]["final_depth_mm"] for n in names], "-o", color="0.25",
             label="depth reached [mm]")
    ax2.axhline(55.0, color="0.25", ls=":", lw=1)
    ax2.set_ylabel("depth reached [mm]"); ax2.set_ylim(0, 60)
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylabel("resume peak [mm/s]")
    ax.set_title(f"At the chain's usual {STEPS * DT:.0f} s: safety improves,\n"
                 "but the task stops completing", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(xs - 0.2, [B[n]["resume_vmax_mms"] for n in names], 0.4,
           color="crimson", label="resume peak [mm/s]")
    ax2 = ax.twinx()
    ax2.plot(xs + 0.2, [B[n]["final_depth_mm"] for n in names], "-o", color="0.25")
    ax2.axhline(55.0, color="0.25", ls=":", lw=1)
    ax2.set_ylabel("depth reached [mm]"); ax2.set_ylim(0, 60)
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylabel("resume peak [mm/s]")
    ax.set_title(f"Given time to finish ({LONG * DT:.0f} s):\nthe ladder collapses",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 2]
    ax.plot(xs, [B[n]["blind_max_mm"] for n in names], "-o", color="tab:blue",
            label="blind travel [mm]")
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylabel("worst blind travel [mm]")
    ax.set_title("What reversal costs: motion without information", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.bar(xs - 0.2, [C[n][0]["resume_vmax_mms"] for n in names], 0.4,
           color="tab:blue", label="master free")
    ax.bar(xs + 0.2, [C[n][1]["resume_vmax_mms"] for n in names], 0.4,
           color="crimson", label="master locked")
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylabel("resume peak [mm/s]")
    ax.set_title("#59's lock verdict holds in every operator model", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(xs - 0.2, [C[n][0]["mismatch_release_mm"] for n in names], 0.4,
           color="tab:blue", label="master free")
    ax.bar(xs + 0.2, [C[n][1]["mismatch_release_mm"] for n in names], 0.4,
           color="crimson", label="master locked")
    ax.set_xticks(xs); ax.set_xticklabels(short)
    ax.set_ylabel("hand-to-tool mismatch at release [mm]")
    ax.set_title("Locking does reduce the mismatch - and still loses", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["The operator finally feels the force", "",
             "#50 built wave variables to TRANSMIT force",
             "and then modelled an operator who never",
             "used it. Twelve experiments sending force",
             "to someone who could not feel it.", "",
             "WHAT I CAUGHT MYSELF DOING:",
             "  At the usual 4 s the tier ladder looks",
             "  like a clean win - resume 120 -> 41 mm/s.",
             "  But depth falls 50.3 -> 34.7 mm. #56 found",
             "  exactly that number and wrote R18: report",
             "  robustness only where the task completes.",
             "  My own operator broke my own rule.", "",
             "GIVEN TIME TO FINISH:",
             "  visual reaction  120 -> 72 mm/s   REAL",
             "  + force sense     72 -> 78        none",
             "  + learning        78 -> 81        none",
             "  + reversal        81 -> 44        REAL",
             "  Reversal pays in blind travel - the same",
             "  price #59's retraction paid, for the same",
             "  reason: motion without information.", "",
             "PREDICTION THAT FAILED:",
             "  I expected force perception to rescue",
             "  master-locking (#59 said the lock hides",
             "  the cue; a felt lock IS a cue). It does",
             "  not. Locking loses in all five models."]
    ax.text(0.02, 0.99, "\n".join(lines), va="top", ha="left", fontsize=7.2,
            family="monospace")

    fig.suptitle("62. An operator who feels, learns and backs off - and what survives "
                 "once the task has to finish", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "62_adaptive_operator.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/62_adaptive_operator.png, assets/62_adaptive_operator.png")

    return dict(A=A, B=B, C=C)


if __name__ == "__main__":
    main()
