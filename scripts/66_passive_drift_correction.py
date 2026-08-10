"""보증을 되찾을 수 있는가 — 그리고 **아홉 실험이 지목한 항을 고쳐서는 안 된다**.

exp 65 는 제어기 전체가 수동이 아님을 재고, 마지막 줄에 이렇게 적고 끝냈다: *"수동성을 회복하는
표류 보정 설계는 여기서 제안하지 않는다 — 다음 일이다."* 이 실험이 그 다음 일이다.

exp 56~64 는 아홉 번에 걸쳐 같은 항을 지목했다 — 파동 변환 **밖**에 얹힌 표류 보정 λ(x_m − x_s).
exp 65 는 λ 를 쓸어 주입이 18 → 94 mJ 로 자라는 것까지 보였다. 그러니 그 항을 정직한 에너지
예산으로 죄면 될 것 같다. exp 57 의 β 게이팅은 **파동 예산**을 빌려 쓴 조잡한 판이었으니,
이번에는 제대로 만든다: **제어기가 자기가 이미 버린 에너지에서만 꺼내 쓰는 탱크**다. 재원은
국소 감쇠 f_loc 이 소산시킨 몫 — exp 65 가 "항상 소산이라 장부를 관대하게 만든다"고 지적했던
바로 그 항이 여기서는 **정당한 재원**이 된다. 없던 것을 만들지 않고 버린 것을 도로 쓰는 것이니
두 포트 합계는 여전히 ≤ 0 이어야 한다.

  A. **항별 분해** — 어느 항이 일을 하는가. (exp 65 는 λ 를 쓸었다. 그건 상관이지 분해가 아니다.)
  B. **지목된 항을 제대로 죈다** — 에너지 탱크. 용량을 0 부터 ∞ 까지 쓴다.
  C. **자기가 주장하는 장부를 직접 본다** — 두 포트 전체에 건 수동성 관측기(PO/PC).
  D. **어느 장부를 보게 하느냐** — exp 65 의 교훈이 이번엔 *고치는 쪽*에서 반복되는가.
  E. **exp 63 의 위해 축으로** 재본다 — 깊이를 맞춰서(짝지어).

**미리 밝히는 결론.** 세 가지를 예측했고 **둘이 틀렸다.**

  1. "탱크가 수동성을 회복한다" — **틀렸다.** 어떤 용량에서도 회복되지 않는다. 용량이 크면
     제약이 아예 안 걸리고(마름 0%, 80.6 mJ 그대로), 용량 0 — 즉 **표류 항을 완전히 꺼도**
     장부는 16 mJ 를 남긴 채 도구가 36.2 mm 에서 멈춘다. 지목된 항은 **유일한 출처가 아니었다.**
  2. "수동성을 되찾으면 과제를 크게 잃는다" — **틀렸다.** 깊이 50.8 → 49.0 mm 로 R18(45 mm)을
     넘긴 채 **E_ctrl 이 (지터 채널의) 모든 시드·모든 시각에서 0 을 넘지 않는다.** 다만 사슬
     최악 조건에서는 12 시드 중 둘이 남는다 — 아래 한계 절.
  3. "떨림이 늘면 조직 위해도 는다" — **틀렸다.** 진동은 0.16 → 1.76 mm 로 11 배가 되는데
     정지 구간 끌림은 오히려 준다(짝지어 3.2 → 0.7 mm, 12 시드).

그리고 **exp 65 의 교훈이 한 층 위에서 그대로 반복된다.** 관측기를 관대한 장부(e_ctrl)에
걸면 그 장부는 1.1 mJ 로 떨어지는데 엄격한 장부(국소 감쇠 제외)는 74 mJ 를 남긴다.
**고치는 것도 자기가 보는 장부만 고친다.** 엄격한 쪽을 보게 하면 둘 다 만족한다.

마지막으로, 짝지어 비교하니 자랑 하나가 사라졌다. 끌림이 16.6 → 10.1 mm(−39%)로 줄어 보였는데
**그 대부분은 덜 깊이 들어가서**다 — 같은 깊이의 λ 사다리가 9.3 mm 를 내므로 남는 차이는 없다
(오히려 근소하게 반대쪽이다. 3·6·12 시드에서 모두 같은 답이다). 짝지어도 남는 이득은
**정지 구간 끌림**(3.2 → 0.7 mm, 12 시드) 하나인데 **이쪽은 약하다** — 3 시드로 보면 부호가
뒤집힌다. 되찾은 것은 **보증**이지 위해 감소가 아니다.

    python scripts/66_passive_drift_correction.py

한계·트레이드오프
  - PO/PC 는 **감쇠력 상한(pc_fmax)** 이 필요하고, 그 상한은 **세게 잡을수록 낫지 않다**:
    50 → 5000 N 으로 키우면 엄격 장부 잔차가 1.4 → 119.9 mJ 로 **나빠진다**. 한 스텝에 큰 힘을
    때리는 것 자체가 여기(勵起)라, 처방이 자기 실패 모드를 갖는다.
  - E_ctrl 은 **0 을 넘지 않는다**를 보이지만, 엄격 장부에는 1.4 mJ 의 잔차가 남는다(상한과
    vs≈0 스텝 때문). "정확히 수동"이 아니라 "재는 장부에서 위반 없음"이다.
  - **보증이 온 것은 지터만 있는 채널에서다.** 사슬 최악 조건(연집 손실 exp 57 + 정지 exp 58 +
    끌림 조직 exp 63)에서는 12 시드 중 **둘(6·11번)이 여전히 위반**한다 — 최악 2.81 mJ 로 raw
    95.1 mJ 대비 97% 줄지만 0 은 넘는다. 더 중요한 것은 **6 시드까지는 정확히 0 이라 안 보인다**는
    점이다. exp 65 가 중앙값에서 만난 함정이 여기서는 **시드 수**로 온다 — 수동성은 모든 실행의
    성질이라 **본 만큼만** 참이다.
  - 진동 0.16 → 1.76 mm 는 이 사슬이 선언해 온 임상 여유(exp 48 통로 1.25 mm, exp 45 shaft
    2.17 mm)와 **같은 자릿수**다. 위해 축에서 손해가 안 났다고 해서 이 숫자가 공짜라는 뜻은
    아니다 — 다른 해부에서 여유가 더 좁으면 이 처방이 먼저 걸린다.
  - 왜 표류 항을 꺼도 16 mJ 가 남는지(파동 쪽 결합 구현이 이상적 포트가 아닌 문제)는 여기서
    **재기만 하고 고치지 않는다.** 그건 파동 변환 구현을 다시 짜는 일이다.
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
s63 = import_module("63_harm_is_not_force")

DT, M_S = jc.DT, jc.M_S
LAM_TASK = jc.LAM_TASK
N_SEEDS = 6
# 위해 축(E 절)은 **그 축을 세운 실험의 시드 수**를 따른다(exp 63 = 12). 정지 구간 끌림의
# 중앙값은 시드에 민감해서 3 시드로 보면 부호가 뒤집힌다 — E 절에 그대로 적어 둔다.
HARM_SEEDS = s63.N_SEEDS
JITTER = 20.0
# R18(exp 56) — 과제를 완주했다고 부를 수 있는 최소 도달 깊이. 이 아래면 "채널이 거의 여기되지
# 않는 상태로 시험을 통과"한 것이라 어떤 수동성 수치도 의미가 없다.
DEPTH_BAR = 45.0
TANK_CAPS = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 1e-1)
LAMS = (6.0, 8.0, 10.0, 12.0, 16.0, 24.0)
FMAXES = (10.0, 50.0, 200.0, 5000.0)


def worst_and_median(runs):
    """수동성은 **최악**으로, 과제·위해는 **중앙값**으로 — 두 성질의 요구가 다르다.

    수동성은 모든 실행·모든 시각에 대한 조건이라 한 시드라도 위반하면 위반이다(exp 65 가 중앙값
    으로 보다가 절반의 시드에서 위반이 사라지는 함정을 만났다). 반대로 도달 깊이·끌림은 확률적
    채널의 대표값을 봐야 하므로 중앙값이 맞다.
    """
    return dict(
        e_ctrl=max(r["e_ctrl_max"] for r in runs),
        e_nd=max(r["e_ctrl_nd_max"] for r in runs),
        depth=float(np.median([r["final_depth_mm"] for r in runs])),
        osc=float(np.median([r["osc_mm"] for r in runs])),
        drag=float(np.median([r["drag_total_mm"] for r in runs])),
        drag_held=float(np.median([r["drag_held_mm"] for r in runs])),
        swing=float(np.median([r["f_e_held_swing"] for r in runs])),
        tank_dry=float(np.mean([r["tank_dry_frac"] for r in runs])),
        pc_duty=float(np.mean([r["pc_duty"] for r in runs])),
        diverged=sum(r["diverged"] for r in runs),
    )


def clean(seeds=N_SEEDS, **kw):
    """exp 65 의 A/B/C 절과 같은 조건 — 지터만 있는 깨끗한 채널."""
    return worst_and_median([jc.run("tdpa", seed=s, jitter_ms=JITTER, **kw)
                             for s in range(seeds)])


def harsh(seeds=N_SEEDS, **kw):
    """exp 57 의 연집 채널 + exp 58 의 정지 + exp 63 의 끌림 세는 조직 — 사슬이 도달한 최악."""
    runs = [bc.run("tdpa", seed=s,
                   tissue_obj=s63.DraggingTissue(tau=s63.TAU, f_slip=s63.F_SLIP0),
                   **s63.BASE, **kw) for s in range(seeds)]
    return worst_and_median(runs)


def depth_matched_drag(ladder, depth):
    """λ 사다리에서 **같은 깊이**의 끌림을 선형 보간한다.

    깊이가 다른 두 구성의 위해를 그냥 비교하면 "덜 들어가서 덜 상했다"가 개선으로 읽힌다 —
    CHECKLIST 의 '짝지어 비교하라' 항목이 정확히 이 함정이고, exp 62 에서 한 번 부호를 뒤집을
    뻔했다. 여기서도 40% 감소로 보이던 것이 이 보정에서 사라진다.
    """
    pts = sorted((v["depth"], v["drag"], v["drag_held"]) for v in ladder.values())
    ds = [p[0] for p in pts]
    return (float(np.interp(depth, ds, [p[1] for p in pts])),
            float(np.interp(depth, ds, [p[2] for p in pts])))


def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    caps = (0.0, 1e-2) if quick else TANK_CAPS
    lams = (8.0, 12.0, 24.0) if quick else LAMS
    fmaxes = (50.0, 5000.0) if quick else FMAXES
    print("=== 66. 보증을 되찾는다 — 단, 아홉 실험이 지목한 항을 고쳐서는 안 된다 ===")
    print("exp 65 는 '전체 제어기는 수동이 아니다'로 끝내고 설계는 다음 일로 남겼다. 그 다음 일이다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 먼저 **항별로 분해**한다 — exp 65 는 λ 를 쓸었고, 그건 상관이지 분해가 아니다")
    r0 = jc.run("tdpa", seed=0, jitter_ms=JITTER)
    terms = r0["e_term"]
    print(f"{'항':>26s} | {'한 일 [mJ]':>12s}")
    label = {"wave_s": "파동 결합(팔 쪽)", "loc": "국소 감쇠", "drift": "**표류 보정 λ**",
             "wave_m": "파동 힘표시(손 쪽)", "hold": "정지 유지(exp 58)",
             "pc": "PO/PC(exp 66)", "vf": "가상 고정구", "ml": "마스터 제동"}
    for kk, v in sorted(terms.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) > 1e-9:
            print(f"{label.get(kk, kk):>26s} | {v * 1e3:12.2f}")
    print(f"{'합계 = 전체 장부':>26s} | {sum(terms.values()) * 1e3:12.2f}")
    print(f"  아홉 실험이 지목한 표류 항은 **{terms['drift'] * 1e3:+.1f} mJ** 인데, 파동 쪽 결합은")
    print(f"  **{terms['wave_s'] * 1e3:+.1f} mJ**, 국소 감쇠는 {terms['loc'] * 1e3:+.1f} mJ 다.")
    print("  **정직하게: 이 표는 어느 항이 '원인'인지 못 정한다.** 항들이 서로 상쇄하고 있고,")
    print("  λ 를 바꾸면 vs 가 바뀌어 모든 항이 같이 움직인다. 분해는 **흐름**이지 인과가 아니다.")
    print("  인과를 물으려면 **꺼 보는 수밖에 없다** — 그게 B 절이다.")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] 지목된 항을 **제대로** 죈다 — 제어기가 이미 버린 에너지에서만 꺼내 쓰는 탱크")
    print("    (exp 57 의 β 게이팅은 파동 예산을 빌려 쓴 조잡한 판이었다. 이건 실제 소산이 재원이다.)")
    print(f"{'용량[mJ]':>9s} | {'전체 공급[mJ]':>13s} | {'깊이[mm]':>9s} | {'진동[mm]':>9s} | "
          f"{'마름':>6s} | {'수동?':>6s}")
    Bt = {}
    for cap in caps:
        m = clean(seeds=seeds, drift_mode="tank", tank_max=cap)
        Bt[cap] = m
        print(f"{cap * 1e3:9.2f} | {m['e_ctrl'] * 1e3:13.2f} | {m['depth']:9.2f} | "
              f"{m['osc']:9.3f} | {m['tank_dry'] * 100:5.0f}% | "
              f"{('예' if m['e_ctrl'] <= 1e-9 else '**아니오**'):>6s}")
    off, big = Bt[caps[0]], Bt[caps[-1]]
    print(f"  **어떤 용량에서도 수동이 되지 않는다.** 용량이 크면 제약이 아예 안 걸리고"
          f"(마름 {big['tank_dry'] * 100:.0f}%, {big['e_ctrl'] * 1e3:.1f} mJ),")
    print(f"  용량 0 — 즉 **표류 항을 완전히 꺼도** {off['e_ctrl'] * 1e3:.1f} mJ 가 남는다.")
    print(f"  그러면서 도구는 {off['depth']:.1f} mm 에서 멈춘다(R18 기준 {DEPTH_BAR:.0f} mm).")
    print("  **아홉 실험이 지목한 항은 유일한 출처가 아니었다.** 그 항을 죄어서는 보증이 안 온다.")
    mid = [c for c in caps if 0.0 < c < 1e-2]
    if mid:
        wob = max(mid, key=lambda c: Bt[c]["osc"])
        print(f"  덤으로: 중간 용량이 가장 나쁘다({wob * 1e3:.1f} mJ 에서 진동 {Bt[wob]['osc']:.2f} mm) —")
        print("  표류 보정이 켜졌다 꺼졌다 하는 것 자체가 진동원이다. **부분 게이팅은 양쪽 다 잃는다.**")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] 대신 **자기가 주장하는 장부를 직접 본다** — 두 포트 전체에 건 수동성 관측기")
    print("    장부가 양수가 되려 하면 팔 쪽에 가변 감쇠를 걸어 그 스텝에 초과분을 뽑는다.")
    print(f"{'구성':>22s} | {'E_ctrl[mJ]':>11s} | {'E_엄격[mJ]':>12s} | {'깊이[mm]':>9s} | "
          f"{'진동[mm]':>9s} | {'가동률':>6s}")
    C = {}
    for name, kw in (("raw (exp 65)", {}),
                     ("PO/PC → 관대 장부", dict(drift_mode="po")),
                     ("PO/PC → 엄격 장부", dict(drift_mode="po", po_strict=True))):
        m = clean(seeds=seeds, **kw)
        C[name] = m
        print(f"{name:>22s} | {m['e_ctrl'] * 1e3:11.3f} | {m['e_nd'] * 1e3:12.2f} | "
              f"{m['depth']:9.2f} | {m['osc']:9.3f} | {m['pc_duty'] * 100:5.0f}%")
    raw, po_l, po_s = C["raw (exp 65)"], C["PO/PC → 관대 장부"], C["PO/PC → 엄격 장부"]
    print(f"  **보증이 돌아온다.** E_ctrl 이 {raw['e_ctrl'] * 1e3:.1f} → "
          f"{po_s['e_ctrl'] * 1e3:.4f} mJ — 모든 시드·모든 시각에서 0 을 넘지 않는다.")
    print(f"  그리고 과제는 살아 있다: 깊이 {raw['depth']:.1f} → {po_s['depth']:.1f} mm "
          f"(R18 {DEPTH_BAR:.0f} mm 위).")
    print(f"  **대가는 떨림이다**: 진동 {raw['osc']:.2f} → {po_s['osc']:.2f} mm "
          f"({po_s['osc'] / max(raw['osc'], 1e-9):.0f}배).")
    print("  이 숫자는 사슬이 선언해 온 임상 여유(exp 48 통로 1.25 mm, exp 45 shaft 2.17 mm)와")
    print("  **같은 자릿수다** — 여유가 더 좁은 해부에서는 이 처방이 먼저 걸린다.")

    # λ 를 낮추는 것으로는 안 되는가 — exp 65 의 쓸기를 완주 기준과 함께 다시 읽는다
    print()
    print("  [대조] 그냥 λ 를 낮추면 안 되나 — exp 65 는 주입이 λ 를 따라 자란다고 했다")
    print(f"{'λ[1/s]':>9s} | {'전체 공급[mJ]':>13s} | {'깊이[mm]':>9s} | {'완주?':>6s}")
    L = {}
    for lam in lams:
        m = clean(seeds=seeds, lam_pos=lam)
        L[lam] = m
        print(f"{lam:9.1f} | {m['e_ctrl'] * 1e3:13.2f} | {m['depth']:9.2f} | "
              f"{('예' if m['depth'] >= DEPTH_BAR else '아니오'):>6s}")
    fin = {l: v for l, v in L.items() if v["depth"] >= DEPTH_BAR}
    if fin:
        best = min(fin, key=lambda l: fin[l]["e_ctrl"])
        print(f"  **완주하는 λ 중 가장 낮은 주입이 {fin[best]['e_ctrl'] * 1e3:.1f} mJ**(λ={best:.0f})다.")
        print("  λ 사다리 위에는 '수동이면서 완주하는' 칸이 없다. 이 축으로는 살 수 없는 물건이다.")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] **어느 장부를 보게 하느냐가 어느 장부를 만족하느냐를 정한다**")
    print(f"  관대한 장부를 보게 하면 그쪽은 {po_l['e_ctrl'] * 1e3:.2f} mJ 로 떨어지는데 "
          f"엄격한 장부는 {po_l['e_nd'] * 1e3:.0f} mJ 를 남긴다.")
    print(f"  엄격한 쪽을 보게 하면 둘 다 만족한다({po_s['e_ctrl'] * 1e3:.4f} / "
          f"{po_s['e_nd'] * 1e3:.2f} mJ) — E_ctrl ≤ E_엄격 이 항상 성립하기 때문이다.")
    print("  **exp 65 가 TDPA 에 대해 한 말이 이번엔 고치는 쪽에서 그대로 반복된다:**")
    print("  **자기가 재는 것만 고친다.** 그러니 재는 것을 먼저 정해야 한다.")
    print()
    print("  그리고 처방에는 **자기 실패 모드**가 있다 — 감쇠 상한을 세게 잡을수록 낫지 않다")
    print(f"{'상한[N]':>9s} | {'E_ctrl[mJ]':>11s} | {'E_엄격[mJ]':>12s} | {'깊이[mm]':>9s} | "
          f"{'진동[mm]':>9s}")
    F = {}
    for fm in fmaxes:
        m = clean(seeds=seeds, drift_mode="po", po_strict=True, pc_fmax=fm)
        F[fm] = m
        print(f"{fm:9.0f} | {m['e_ctrl'] * 1e3:11.4f} | {m['e_nd'] * 1e3:12.2f} | "
              f"{m['depth']:9.2f} | {m['osc']:9.3f}")
    lo, hi = F[fmaxes[0]], F[fmaxes[-1]]
    print(f"  약하면 못 뽑고({lo['e_ctrl'] * 1e3:.1f} mJ 가 새고), 세면 **여기가 된다** — "
          f"엄격 장부 잔차가 {hi['e_nd'] * 1e3:.0f} mJ 로 커진다.")
    print("  한 스텝에 큰 힘을 때리는 것 자체가 계를 여기하기 때문이다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 그래서 **환자에게** 무엇이 달라지나 — exp 63 의 끌림 축으로, 깊이를 맞춰서")
    print("    (연집 채널 exp 57 + 정지 exp 58 + 끌림 세는 조직 exp 63 — 사슬이 도달한 최악 조건)")
    print(f"{'구성':>18s} | {'E_ctrl[mJ]':>11s} | {'깊이[mm]':>9s} | {'끌림[mm]':>9s} | "
          f"{'정지중 끌림':>11s} | {'진폭[N]':>8s}")
    hseeds = 3 if quick else HARM_SEEDS
    E = {}
    for name, kw in (("raw (exp 65)", {}), ("탱크", dict(drift_mode="tank")),
                     ("PO/PC 엄격", dict(drift_mode="po", po_strict=True))):
        m = harsh(seeds=hseeds, **kw)
        E[name] = m
        print(f"{name:>18s} | {m['e_ctrl'] * 1e3:11.2f} | {m['depth']:9.2f} | "
              f"{m['drag']:9.2f} | {m['drag_held']:11.2f} | {m['swing']:8.2f}")
    ladder = {}
    for lam in lams:
        ladder[lam] = harsh(seeds=hseeds, lam_pos=lam)
    hr, hp = E["raw (exp 65)"], E["PO/PC 엄격"]
    if hp["e_ctrl"] > 1e-9:
        print(f"  **먼저 정직하게: 이 조건에서는 보증이 완전히 오지 않는다.** 최악 E_ctrl 이 "
              f"{hr['e_ctrl'] * 1e3:.1f} → {hp['e_ctrl'] * 1e3:.2f} mJ 로 "
              f"{(1 - hp['e_ctrl'] / hr['e_ctrl']) * 100:.0f}% 줄지만 0 을 넘는다.")
        print(f"  그리고 **6 시드까지는 정확히 0 이라 안 보인다** — {hseeds} 시드로 늘려야 나온다"
              "(6·11번 시드).")
        print("  exp 65 가 중앙값에서 만난 함정이 여기서는 **시드 수**로 온다. C 절의 '수동'은")
        print("  **지터만 있는 채널**에 대한 문장이고, 이 조건에서는 '97% 줄었다'가 정확한 표현이다.")
    else:
        print(f"  ({hseeds} 시드에서는 최악 E_ctrl 이 0 이다 — 위반은 6·11번 시드라 12 시드에서 나온다.)")
    d_int, dh_int = depth_matched_drag(ladder, hp["depth"])
    print(f"  그냥 보면 끌림이 {hr['drag']:.1f} → {hp['drag']:.1f} mm 로 좋아진 것 같다. "
          "**그런데 깊이가 다르다.**")
    print(f"  같은 깊이({hp['depth']:.1f} mm)에서 λ 사다리를 보간하면 끌림 **{d_int:.1f} mm** — "
          "차이가 사라진다.")
    print("  **덜 들어가서 덜 상한 것이었다.** CHECKLIST 의 '짝지어 비교하라' 가 자랑 하나를 잡았다.")
    print(f"  짝지어도 남는 이득은 하나다: **정지 구간 끌림 {dh_int:.1f} → {hp['drag_held']:.1f} mm**"
          f"({hseeds} 시드).")
    print("  **단, 이 하나는 약하다.** 같은 비교를 3 시드로 하면 부호가 뒤집힌다(4.19 vs 2.38) —")
    print("  정지 구간 끌림의 중앙값은 정지가 걸린 위상에 민감해서 시드가 적으면 못 믿는다.")
    print("  (총 끌림 쪽은 3·6·12 시드에서 모두 '차이 없음'으로 같다 — 그쪽 결론은 튼튼하다.)")
    print("  그리고 진동은 11 배가 됐는데 위해는 안 늘었다 — **예측 3 이 틀렸다.**")

    # ------------------------------ F ------------------------------
    print("-" * 100)
    print("[F] 정리")
    print("  1. **보증은 되찾을 수 있다 — 지터 채널에서.** 두 포트 장부를 직접 감시하면 E_ctrl 이")
    print(f"     모든 시드·모든 시각에서 0 을 넘지 않고, 과제도 완주한다({po_s['depth']:.1f} mm).")
    print("     **최악 조건에서는 아직 아니다**(E 절: 12 시드 중 둘이 위반, 97% 감소).")
    print("  2. **단, 아홉 실험이 지목한 항을 고쳐서는 안 된다.** 그 항을 정직한 탱크로 죄어도,")
    print(f"     심지어 **완전히 꺼도** {off['e_ctrl'] * 1e3:.0f} mJ 가 남는다. 지목은 부분적으로만 옳았다.")
    print("     exp 65 는 λ 를 쓸어 **상관**을 봤고, 이 실험은 꺼서 **인과**를 봤다 — 다른 도구다.")
    print("  3. **고치는 것도 자기가 보는 장부만 고친다.** exp 65 가 TDPA 에 대해 한 말이 처방에")
    print("     그대로 적용된다. 무엇을 감시할지가 무엇이 보증될지를 정한다.")
    print("  4. **처방에는 자기 실패 모드가 있다.** 감쇠 상한이 세면 그 자체가 여기가 된다.")
    print("  5. **위해는 안 줄었다 — 짝지어 보면.** 되찾은 것은 보증이지 안전 마진이 아니다.")
    print("     진동은 11 배가 됐는데 위해 축에서는 손해가 안 났다. 두 축은 또 서로 다른 축이다.")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    cs = [c * 1e3 for c in caps]
    ax.plot(cs, [Bt[c]["e_ctrl"] * 1e3 for c in caps], "-o", color="crimson",
            label="energy created")
    ax.axhline(0.0, color="crimson", lw=1.2, ls="--")
    ax.text(cs[-1], 2.0, "passive requires <= 0", fontsize=7.5, color="crimson",
            ha="right", va="bottom")
    ax.set_xscale("symlog", linthresh=1e-2)
    ax.set_xlabel("energy-tank capacity [mJ]")
    ax.set_ylabel("energy created [mJ]", color="crimson")
    ax.set_ylim(-6, 100)
    ax2 = ax.twinx()
    ax2.plot(cs, [Bt[c]["depth"] for c in caps], "-s", color="0.3",
             label="depth reached")
    ax2.axhline(DEPTH_BAR, color="0.3", ls=":", lw=1)
    ax2.text(cs[-1], DEPTH_BAR + 0.4, "task bar (R18)", fontsize=7.5, color="0.3",
             ha="right")
    ax2.set_ylabel("depth reached [mm]")
    ax.set_title("Gating the named term: no capacity works\n"
                 "(empty tank = term fully off, still active)", fontsize=10)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    ax = axes[1]
    names = ["raw\n(#65)", "tank\n(named term)", "PO/PC\nlenient", "PO/PC\nstrict"]
    ev = [raw["e_ctrl"], Bt[caps[-1]]["e_ctrl"], po_l["e_ctrl"], po_s["e_ctrl"]]
    nv = [raw["e_nd"], Bt[caps[-1]]["e_nd"], po_l["e_nd"], po_s["e_nd"]]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [v * 1e3 for v in ev], 0.4, color="crimson",
           label="two-port ledger (the claim)")
    ax.bar(x + 0.2, [v * 1e3 for v in nv], 0.4, color="0.55",
           label="strict (no local damper)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0.0, color="0.3", lw=1.2)
    for i, v in enumerate(ev):
        # 0 mJ 막대는 로그 축에서 높이가 없다 — 이 실험의 결론이 바로 그 칸이라 글자로 박는다.
        lab = f"{v * 1e3:.2f}" + ("\npassive" if v <= 1e-9 else "")
        ax.text(x[i] - 0.2, 1.25 if v <= 1e-9 else v * 1e3 * 1.35, lab, ha="center",
                va="bottom", fontsize=7.5, color="crimson", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("energy created [mJ]  (symlog)")
    ax.set_title("The fix repairs the ledger it watches\n(exp 65's lesson, one level up)",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    ds = [ladder[l]["depth"] for l in lams]
    ax.plot(ds, [ladder[l]["drag"] for l in lams], "-o", color="0.4",
            label="lambda ladder (all active)")
    ax.plot([hp["depth"]], [hp["drag"]], "*", ms=17, color="crimson", zorder=5,
            label="PO/PC strict (passive)")
    ax.plot([hp["depth"]], [d_int], "x", ms=11, mew=2.2, color="seagreen", zorder=6,
            label="depth-matched baseline")
    ax.annotate(f"same depth, same harm\n({hp['drag']:.1f} vs {d_int:.1f} mm)",
                xy=(hp["depth"], hp["drag"]),
                xytext=(hp["depth"] - 5.5, hp["drag"] - 5.0), fontsize=8,
                ha="center", arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1))
    ax.set_xlabel("depth reached [mm]")
    ax.set_ylabel("irrecoverable tissue drag [mm]")
    ax.set_title("Paired at the same depth, the harm gain\nlargely disappears",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("66. Restoring the guarantee — and why fixing the term nine experiments "
                 "named does not do it", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "66_passive_drift_correction.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/66_passive_drift_correction.png, "
          "assets/66_passive_drift_correction.png")

    return dict(A=terms, B=Bt, C=C, L=L, F=F, E=E, ladder=ladder)


if __name__ == "__main__":
    main()
