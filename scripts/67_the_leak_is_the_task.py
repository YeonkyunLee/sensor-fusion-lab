"""남은 16 mJ 의 출처 — **구현 결함이 아니라 침투의 가격이었다**.

exp 66 은 표류 항을 죄어서는 보증이 안 온다는 것까지 보이고, 남은 잔차에 대해 이렇게 적고
끝냈다: *"왜 표류 항을 꺼도 16 mJ 가 남는지(파동 쪽 결합 구현이 이상적 포트가 아닌 문제)는
여기서 **재기만 하고 고치지 않는다.** 그건 파동 변환 구현을 다시 짜는 일이다."*

**그 예보가 틀렸다.** 다시 짤 것이 없다. 파동 변환은 멀쩡하고, **팔 쪽 결합 이득이 파동
임피던스의 6 배**였을 뿐이다(D_S = 60 vs b = 10). 이득을 임피던스에 맞추면 표류 항이 없는
제어기는 **정확히 수동**이 된다 — 그리고 도구가 27.3 mm 에서 멈춘다. 그게 이 실험의 전부다:
**주입은 어느 항의 결함이 아니라 조직을 뚫는 데 드는 값**이다.

  A. **지목도 채널도 아니다** — λ 를 0 으로 두고 지터 0 · 손실 0 에서 다시 잰다.
  B. **결합 이득 사다리** — 이득을 파동 임피던스까지 내리면 어디서 수동이 되는가.
  C. **포트를 대수적으로 닫아 본다** — w = u − √(2b)·v. 이걸로 사는가.
  D. **(이득, λ) 2 차원 격자** — '수동이면서 완주하는' 칸이 존재하는가.
  E. **그래서 exp 66 의 관측기는 무엇이었나** — 교환곡선 위의 점인가 밖의 점인가.
  F. **남은 2/12 는 무엇인가** — 이득을 옮기면 닫히는가.

**미리 밝히는 결론.** 세 가지를 예측했고 **둘이 틀렸다.**

  1. "포트를 대수적으로 닫으면(w = u − √(2b)·v) 잔차는 사라지지만 손에 표시되는 힘이 조직과
     무관해진다" — **절반만 맞았다.** 투명성은 예측대로 팔렸다(힘 오차 0.41 → 2.43 N, 6 배).
     그런데 **보증을 사지도 못했다**: 장부가 16.3 → 46.0 mJ 로 오히려 나빠진다.
  2. "포트 불일치분을 exp 66 의 PO/PC 재원으로 넘기면 최악 조건의 2/12 위반이 닫힌다" —
     **틀렸다.** 운전점을 바꿔도 **같은 시드 6·11 번**이 세 운전점 전부에서 그대로 위반한다.
     이득의 문제가 아니었다.
  3. "결합 이득을 파동 임피던스 b 에 맞추면 잔차가 크게 준다" — **맞았다.** 크게 주는 정도가
     아니라 **정확히 0** 이 된다(표류 항이 없을 때). 다만 그 이득에서는 도구가 27.3 mm 에서
     멈춘다 — 그것만으로는 살 수 없는 물건이다.

그리고 **이 실험을 시작하게 만든 계측기가 원인이 아니었다.** 포트 항등식 ½(u² − w²) = F·v 의
잔차를 재는 계측을 넣고 출발했는데, d_s = 5 에서 그 잔차가 10.3 mJ 인 동안 **장부는 정확히
0** 이다. 실재하는 양을 쟀지만 **인과는 아니었다** — exp 66 의 "분해는 흐름이지 인과가 아니다"
가 한 층 아래에서 그대로 반복된다.

그리고 **exp 66 이 대가를 하나 덜 셌다.** 관측기의 값은 떨림만이 아니다 — 손과 도구의 위치
오차가 2.00 → **10.01 mm** 로 5 배가 된다. exp 66 은 도달 깊이만 보고 "과제는 살아 있다"고
적었는데, 깊이는 살아 있어도 **술자가 명령한 자리와 도구가 있는 자리가 1 cm 어긋난다.**

덤으로, **1 차원으로만 쓸어 온 습관이 더 싼 모서리를 가리고 있었다.** 사슬이 열여섯 실험 동안
써 온 운전점 (d_s, λ) = (60, 24) 옆에 **같은 깊이에서 주입이 37% 싼** (20, 48) 이 있다.
아무도 못 본 이유는 단순하다 — **매번 한 손잡이만 쓸었다.**

    python scripts/67_the_leak_is_the_task.py

한계·트레이드오프
  - **"수동"은 여전히 재는 장부에서의 위반 없음**이다(exp 66 의 한계 그대로). 그리고 이 실험의
    A~E 절은 **지터만 있는 채널**이다 — 최악 조건은 F 절에서만 본다.
  - **값싼 모서리는 공짜가 아니다.** (20, 48) 은 주입을 80.6 → 50.8 mJ 로 줄이는 대신 진동을
    0.16 → 0.48 mm, 위치 오차를 2.00 → 2.40 mm 로 문다. "더 싸다"는 **주입 축에서만** 참이다.
  - **D 절의 '그런 칸은 없다'는 격자 위에서의 진술**이다. 7×6 = 42 칸을 4 시드로 봤고, 칸
    사이는 안 봤다. 격자를 촘촘히 하면 경계 근처에 칸이 있을 수 있다 — 다만 ρ(깊이, 주입)
    = 0.92 라 있어도 좁다.
  - **F 절은 원인을 못 짚는다.** 위반 시드가 관측기 가동률 하위 꼬리(0.663·0.669)에 있는 것은
    맞는데, **시드 5(0.664)가 반례**라 그 스칼라 하나로는 어느 시드가 깨질지 못 맞힌다.
    언제 못 뽑는 스텝이 수요와 겹치는가는 시계열 문제이고 여기서 열어 둔다.
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

B_WAVE = jc.B_WAVE          # 파동 임피던스 — 이 실험의 주인공
D_S_CHAIN = jc.D_S          # 사슬이 써 온 결합 이득(= 6·b)
LAM_CHAIN = jc.LAM_TASK     # exp 56 이 과제를 완주시키려고 올린 표류 이득
JITTER = 20.0
DEPTH_BAR = 45.0            # R18(exp 56)
N_SEEDS = 6
GRID_SEEDS = 4              # 격자는 42 칸이라 시드를 낮춘다 — D 절 한계에 적어 둔다
HARM_SEEDS = s63.N_SEEDS

DS_LADDER = (5.0, 10.0, 20.0, 40.0, 60.0, 120.0)
GRID_DS = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 150.0)
GRID_LAMS = (0.0, 3.0, 6.0, 12.0, 24.0, 48.0)


def clean(seeds=N_SEEDS, **kw):
    """exp 65·66 의 A~C 절과 같은 조건 — 지터만 있는 채널. 수동성은 최악, 나머지는 중앙값."""
    rs = [jc.run("tdpa", seed=s, jitter_ms=JITTER, **kw) for s in range(seeds)]
    return dict(
        e_ctrl=max(r["e_ctrl_max"] for r in rs),
        e_port=max(abs(r["e_port"]) for r in rs),
        depth=float(np.median([r["final_depth_mm"] for r in rs])),
        osc=float(np.median([r["osc_mm"] for r in rs])),
        ferr=float(np.median([r["force_err_N"] for r in rs])),
        perr=float(np.median([r["pos_err_mm"] for r in rs])),
    )


def harsh_seeds(seeds=HARM_SEEDS, **kw):
    """exp 57 연집 채널 + exp 58 정지 + exp 63 끌림 조직 — 사슬이 도달한 최악. 시드별로 돌려준다."""
    return [bc.run("tdpa", seed=s,
                   tissue_obj=s63.DraggingTissue(tau=s63.TAU, f_slip=s63.F_SLIP0),
                   **s63.BASE, **kw) for s in range(seeds)]


def scan(ds=GRID_DS, lams=GRID_LAMS, seeds=GRID_SEEDS):
    """(결합 이득, 표류 이득) 격자. **두 손잡이를 같이 움직이는 것이 이 실험의 방법이다.**

    exp 50~66 은 매번 한 손잡이만 쓸었다(λ 사다리, 용량 사다리, 상한 사다리). 그래서
    "수동이면서 완주하는 칸이 없다"를 아무도 **격자 위에서** 확인한 적이 없고, 동시에 더 싼
    모서리도 아무도 못 봤다. 둘 다 2 차원에서만 보인다.
    """
    out = {}
    for g in ds:
        for lam in lams:
            rs = [jc.run("tdpa", seed=s, jitter_ms=JITTER, d_s=g, lam_pos=lam)
                  for s in range(seeds)]
            out[(g, lam)] = dict(
                e_ctrl=max(r["e_ctrl_max"] for r in rs),
                depth=float(np.median([r["final_depth_mm"] for r in rs])),
                osc=float(np.median([r["osc_mm"] for r in rs])),
            )
    return out


def spearman(a, b):
    """순위 상관 — scipy 없이. exp 63 이 지표 축을 볼 때 쓴 것과 같은 통계다."""
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum()))


def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    gseeds = 1 if quick else GRID_SEEDS
    ds_lad = (10.0, 60.0) if quick else DS_LADDER
    g_ds = (10.0, 60.0, 150.0) if quick else GRID_DS
    g_lams = (0.0, 24.0) if quick else GRID_LAMS
    hseeds = 3 if quick else HARM_SEEDS

    print("=== 67. 남은 16 mJ 은 구현 결함이 아니라 **침투의 가격**이었다 ===")
    print(f"파동 임피던스 b = {B_WAVE:.0f}, 사슬이 써 온 결합 이득 D_S = {D_S_CHAIN:.0f} "
          f"(= {D_S_CHAIN / B_WAVE:.0f}·b) — 여기서 시작한다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 먼저 **지목도 채널도 아님**을 확인한다 — λ 를 0 으로 두고 지터·손실도 0 으로")
    a_off = jc.run("tdpa", seed=0, jitter_ms=0.0, lam_pos=0.0)
    a_jit = jc.run("tdpa", seed=0, jitter_ms=JITTER, lam_pos=0.0)
    print(f"{'조건':>28s} | {'E_ctrl[mJ]':>11s} | {'파동 장부':>10s} | {'깊이[mm]':>9s}")
    for name, r in (("λ=0, 지터 0, 손실 0", a_off), (f"λ=0, 지터 {JITTER:.0f} ms", a_jit)):
        print(f"{name:>28s} | {r['e_ctrl_max'] * 1e3:11.2f} | "
              f"{r['e_min']:10.4f} | {r['final_depth_mm']:9.2f}")
    print(f"  **표류 항을 문자 그대로 없애고 채널을 완벽하게 만들어도 "
          f"{a_off['e_ctrl_max'] * 1e3:.1f} mJ 를 만든다.**")
    print("  그동안 아홉 실험이 보던 파동 블록 장부는 그 내내 정확히 0 이다. exp 66 이 남긴 자리다.")
    print("  (exp 66 은 탱크 용량 0 을 '항을 완전히 껐다'로 적었는데, 정확히는 **주입 방향만**")
    print("   막고 소산 방향은 통과시킨다. 진짜 절제는 이 λ=0 이고, 숫자는 같은 자리를 가리킨다.)")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] **결합 이득 사다리** — 이득을 파동 임피던스까지 내리면 어디서 수동이 되나")
    print(f"    (팔은 f_coup = d_s·vs_cmd 와 f_loc = −d_s·vs 로 명령을 좇는다. "
          f"파동선이 가정한 임피던스는 b = {B_WAVE:.0f} 다.)")
    print(f"{'d_s':>7s} | {'d_s/b':>6s} | {'E_ctrl[mJ]':>11s} | {'포트잔차[mJ]':>13s} | "
          f"{'깊이[mm]':>9s} | {'힘오차[N]':>10s} | {'수동?':>6s}")
    B = {}
    for g in ds_lad:
        m = clean(seeds=seeds, lam_pos=0.0, d_s=g)
        B[g] = m
        print(f"{g:7.0f} | {g / B_WAVE:6.1f} | {m['e_ctrl'] * 1e3:11.3f} | "
              f"{m['e_port'] * 1e3:13.2f} | {m['depth']:9.2f} | {m['ferr']:10.3f} | "
              f"{('예' if m['e_ctrl'] <= 1e-9 else '아니오'):>6s}")
    pas = [g for g in ds_lad if B[g]["e_ctrl"] <= 1e-9]
    if pas:
        gp = max(pas)
        print(f"  **d_s ≤ {gp:.0f} 에서 정확히 수동이다** — 파동 임피던스와 같은 자릿수인 그 지점이다.")
        print(f"  exp 66 은 이 잔차를 '파동 변환 구현을 다시 짜는 일'로 넘겼는데 **다시 짤 게 없다.**")
        print(f"  변환은 멀쩡하고 **이득이 임피던스의 {D_S_CHAIN / B_WAVE:.0f} 배**였을 뿐이다.")
        print(f"  그런데 그 이득에서 도구는 {B[gp]['depth']:.1f} mm 에서 멈춘다"
              f"(R18 {DEPTH_BAR:.0f} mm). **이것만으로는 살 수 없다.**")
    print("  그리고 **포트 항등식 잔차는 장부를 못 설명한다** — 위 표에서 장부가 0 인 칸에도")
    print("  잔차가 남는다. 이 실험을 시작하게 만든 계측기가 **실재하지만 인과는 아니었다.**")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] 그럼 **포트를 대수적으로 닫으면** 되나 — w = u − √(2b)·v 로 두면 항등식은 구성상 참")
    print(f"{'구성':>26s} | {'E_ctrl[mJ]':>11s} | {'포트잔차[mJ]':>13s} | {'깊이[mm]':>9s} | "
          f"{'힘오차[N]':>10s}")
    C = {}
    for name, kw in (("legacy, λ=0", dict(lam_pos=0.0)),
                     ("reflect, λ=0", dict(lam_pos=0.0, port_mode="reflect")),
                     ("legacy, λ=사슬 값", dict()),
                     ("reflect, λ=사슬 값", dict(port_mode="reflect"))):
        m = clean(seeds=seeds, **kw)
        C[name] = m
        print(f"{name:>26s} | {m['e_ctrl'] * 1e3:11.3f} | {m['e_port'] * 1e3:13.2f} | "
              f"{m['depth']:9.2f} | {m['ferr']:10.3f}")
    cl, cr = C["legacy, λ=사슬 값"], C["reflect, λ=사슬 값"]
    print(f"  **예측 1 이 절반 틀렸다.** 투명성은 예측대로 팔린다 — 힘 오차 {cl['ferr']:.2f} → "
          f"{cr['ferr']:.2f} N({cr['ferr'] / max(cl['ferr'], 1e-9):.0f} 배).")
    print("  그런데 **보증을 사지도 못한다**: λ=0 에서 장부가 오히려 나빠진다"
          f"({C['legacy, λ=0']['e_ctrl'] * 1e3:.1f} → {C['reflect, λ=0']['e_ctrl'] * 1e3:.1f} mJ).")
    print("  항등식을 **구성상 참으로 만드는 것**과 계가 실제로 수동인 것은 다른 문제다.")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] **두 손잡이를 같이 움직인다** — (결합 이득, 표류 이득) 격자에 '수동이면서 완주'가 있나")
    print(f"    (exp 50~66 은 매번 한 손잡이만 쓸었다. 그래서 이 질문이 한 번도 안 물어졌다.)")
    G = scan(ds=g_ds, lams=g_lams, seeds=gseeds)
    print(f"{'d_s':>6s} |" + "".join(f"{('λ=%g' % l):>10s}" for l in g_lams))
    for g in g_ds:
        print(f"{g:6.0f} |" + "".join(f"{G[(g, l)]['e_ctrl'] * 1e3:10.2f}" for l in g_lams))
    print("  같은 격자의 도달 깊이 [mm]:")
    print(f"{'d_s':>6s} |" + "".join(f"{('λ=%g' % l):>10s}" for l in g_lams))
    for g in g_ds:
        print(f"{g:6.0f} |" + "".join(f"{G[(g, l)]['depth']:10.2f}" for l in g_lams))
    fin = {k: v for k, v in G.items() if v["depth"] >= DEPTH_BAR}
    both = {k: v for k, v in fin.items() if v["e_ctrl"] <= 1e-9}
    ev = np.array([v["e_ctrl"] * 1e3 for v in G.values()])
    dv = np.array([v["depth"] for v in G.values()])
    rho = spearman(dv, ev)
    print(f"  **수동이면서 완주하는 칸: {len(both)} 개.** 완주하는 칸 {len(fin)} 개의 주입은")
    if fin:
        print(f"  최소 {min(v['e_ctrl'] for v in fin.values()) * 1e3:.1f} mJ 이고, 완주 못 하는 칸은"
              f" 중앙값 {np.median([v['e_ctrl'] * 1e3 for v in G.values() if v['depth'] < DEPTH_BAR]):.1f} mJ 다.")
    print(f"  ρ(깊이, 주입) = **{rho:.2f}** — **주입은 어느 항의 결함이 아니라 침투의 가격이다.**")
    if fin:
        cheap = min(fin, key=lambda k: fin[k]["e_ctrl"])
        print(f"  덤: 1 차원 습관이 가린 모서리가 있다. 사슬의 운전점 "
              f"({D_S_CHAIN:.0f}, {LAM_CHAIN:.0f}) 옆에 **{cheap} 이 있다.**")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 그래서 **exp 66 의 관측기는 무엇이었나** — 교환곡선 위의 점인가, 밖의 점인가")
    print(f"{'구성':>26s} | {'E_ctrl[mJ]':>11s} | {'깊이[mm]':>9s} | {'진동[mm]':>9s} | "
          f"{'위치오차[mm]':>12s}")
    E = {}
    for name, kw in ((f"사슬 운전점 ({D_S_CHAIN:.0f}, {LAM_CHAIN:.0f})", dict()),
                     ("값싼 모서리 (20, 48)", dict(d_s=20.0, lam_pos=48.0)),
                     ("exp 66 관측기(엄격)", dict(drift_mode="po", po_strict=True))):
        m = clean(seeds=seeds, **kw)
        E[name] = m
        print(f"{name:>26s} | {m['e_ctrl'] * 1e3:11.4f} | {m['depth']:9.2f} | "
              f"{m['osc']:9.3f} | {m['perr']:12.3f}")
    po = E["exp 66 관측기(엄격)"]
    chain0 = E[f"사슬 운전점 ({D_S_CHAIN:.0f}, {LAM_CHAIN:.0f})"]
    print(f"  격자 어디에도 없던 칸을 관측기가 준다 — **수동이면서 {po['depth']:.1f} mm**.")
    print("  **exp 66 이 산 것은 교환곡선 위의 좋은 자리가 아니라 곡선 밖의 점이었다.**")
    print(f"  **그리고 값이 exp 66 이 적은 것보다 비싸다.** 떨림 말고 **위치 추종**도 문다 —")
    print(f"  손과 도구의 위치 오차가 {chain0['perr']:.2f} → {po['perr']:.2f} mm "
          f"({po['perr'] / max(chain0['perr'], 1e-9):.0f} 배)다.")
    print("  exp 66 은 도달 깊이만 보고 '과제는 살아 있다'고 적었는데, 깊이는 살아 있어도")
    print("  **술자가 명령한 자리와 도구가 있는 자리가 1 cm 어긋난다.** 대가 하나를 덜 셌다.")

    # ------------------------------ F ------------------------------
    print("-" * 100)
    print("[F] 그럼 exp 66 이 남긴 **최악 조건의 2/12 위반**은 이득으로 닫히나")
    print(f"{'구성':>30s} | {'최악 E_ctrl[mJ]':>15s} | {'위반 시드':>14s} | {'깊이[mm]':>9s}")
    F = {}
    for name, kw in (("관측기 @ 사슬 운전점", dict(drift_mode="po", po_strict=True)),
                     ("관측기 @ 값싼 모서리", dict(drift_mode="po", po_strict=True,
                                             d_s=20.0, lam_pos=48.0)),
                     ("관측기 @ d_s=30", dict(drift_mode="po", po_strict=True,
                                            d_s=30.0, lam_pos=48.0))):
        rs = harsh_seeds(seeds=hseeds, **kw)
        # 이름을 D 절의 ev/dv 와 겹치지 않게 둔다 — 겹쳐서 그림이 한 번 깨졌다.
        e_seed = [r["e_ctrl_max"] * 1e3 for r in rs]
        bad = [i for i, v in enumerate(e_seed) if v > 1e-9]
        F[name] = (max(e_seed), bad, float(np.median([r["final_depth_mm"] for r in rs])),
                   [r["pc_duty"] for r in rs])
        print(f"{name:>30s} | {max(e_seed):15.4f} | {str(bad):>14s} | {F[name][2]:9.2f}")
    sets = [set(v[1]) for v in F.values()]
    if sets and all(s == sets[0] for s in sets) and sets[0]:
        print(f"  **예측 2 가 틀렸다.** 운전점을 바꿔도 **같은 시드 {sorted(sets[0])} 가** 위반한다 —")
        print("  이득의 문제가 아니다. 그 시드들이 무엇이 다른지 봐야 한다.")
    duty = F["관측기 @ 사슬 운전점"][3]
    if duty and sets and sets[0]:
        order = np.argsort(duty)
        rank = {int(i): int(np.where(order == i)[0][0]) + 1 for i in range(len(duty))}
        print(f"  위반 시드의 관측기 **가동률 순위**(작을수록 덜 개입): "
              f"{[(i, rank[i]) for i in sorted(sets[0])]} / {len(duty)}")
        print(f"  하위 꼬리에 있는 것은 맞다. **그런데 그것만으로는 못 맞힌다** — 가동률 "
              f"{min(duty):.3f} 인 시드가 깨끗하다.")
        print("  못 뽑는 스텝(vs≈0·정지 혼합)이 **언제 수요와 겹치는가**는 시계열 문제라 여기서 연다.")

    # ------------------------------ G ------------------------------
    print("-" * 100)
    print("[G] 정리")
    print("  1. **남은 16 mJ 은 구현 결함이 아니었다.** 파동 변환은 멀쩡하고 결합 이득이")
    print(f"     임피던스의 {D_S_CHAIN / B_WAVE:.0f} 배였다. 맞추면 정확히 수동이 되고, 과제를 잃는다.")
    print("     **exp 66 의 예보('파동 변환을 다시 짜는 일')가 틀렸다.**")
    print(f"  2. **주입은 침투의 가격이다.** 격자에서 R18 을 넘는 칸은 전부 비싸고, "
          f"ρ = {rho:.2f} 다.")
    print("     열여섯 실험이 '결함 있는 항'을 찾던 것은 **가격표를 결함으로 읽은 것**이다.")
    print("  3. **그래서 exp 66 의 관측기가 산 것은 곡선 밖의 점**이다 — 이득 두 개로는 못 가는 칸.")
    chain, corner = E[f"사슬 운전점 ({D_S_CHAIN:.0f}, {LAM_CHAIN:.0f})"], E["값싼 모서리 (20, 48)"]
    print(f"  4. **1 차원 습관이 더 싼 모서리를 가렸다.** 같은 깊이"
          f"({chain['depth']:.1f} vs {corner['depth']:.1f} mm)에서 주입이 "
          f"{chain['e_ctrl'] * 1e3:.1f} → {corner['e_ctrl'] * 1e3:.1f} mJ "
          f"({(1 - corner['e_ctrl'] / chain['e_ctrl']) * 100:.0f}% 싸다).")
    print(f"     공짜는 아니다 — 진동 {chain['osc']:.2f} → {corner['osc']:.2f} mm, 위치 오차 "
          f"{chain['perr']:.2f} → {corner['perr']:.2f} mm 를 문다.")
    print("     그래도 **매번 한 손잡이만 쓸면 곡선은 보여도 지형은 안 보인다.**")
    print("  5. **내 계측기가 원인이 아니었다.** 포트 항등식 잔차는 실재하지만 장부와 안 맞는다.")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    gg = list(ds_lad)
    ax.plot(gg, [B[g]["e_ctrl"] * 1e3 for g in gg], "-o", color="crimson",
            label="energy created")
    ax.axhline(0.0, color="crimson", lw=1.2, ls="--")
    ax.axvline(B_WAVE, color="seagreen", lw=1.4, ls=":")
    ax.text(B_WAVE * 1.08, 12.0, "wave impedance b", fontsize=7.5, color="seagreen")
    ax.set_xscale("log")
    ax.set_xlabel("arm-side coupling gain  d_s  [N s/m]")
    ax.set_ylabel("energy created [mJ]", color="crimson")
    ax2 = ax.twinx()
    ax2.plot(gg, [B[g]["depth"] for g in gg], "-s", color="0.3", label="depth reached")
    ax2.axhline(DEPTH_BAR, color="0.3", ls=":", lw=1)
    ax2.text(gg[0], DEPTH_BAR + 0.6, "task bar (R18)", fontsize=7.5, color="0.3")
    ax2.set_ylabel("depth reached [mm]")
    ax.set_title("The residual was never an implementation bug:\n"
                 "the gain was 6x the impedance the line assumes", fontsize=10)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    ax = axes[1]
    M = np.array([[G[(g, l)]["e_ctrl"] * 1e3 for l in g_lams] for g in g_ds])
    Dm = np.array([[G[(g, l)]["depth"] for l in g_lams] for g in g_ds])
    im = ax.imshow(M, origin="lower", aspect="auto", cmap="magma_r",
                   norm=matplotlib.colors.LogNorm(vmin=max(M[M > 0].min(), 1e-2),
                                                  vmax=M.max()))
    ax.set_xticks(range(len(g_lams))); ax.set_xticklabels([f"{l:g}" for l in g_lams])
    ax.set_yticks(range(len(g_ds))); ax.set_yticklabels([f"{g:g}" for g in g_ds])
    ax.set_xlabel("drift gain  lambda"); ax.set_ylabel("coupling gain  d_s")
    ax.contour(Dm, levels=[DEPTH_BAR], colors="deepskyblue", linewidths=2.0)
    ax.text(0.04, 0.94, "above the blue line: task done", transform=ax.transAxes,
            fontsize=7.5, color="deepskyblue", va="top")
    for (g, l), mark, lab in (((D_S_CHAIN, LAM_CHAIN), "o", "chain"),):
        if g in g_ds and l in g_lams:
            ax.plot(g_lams.index(l), g_ds.index(g), mark, ms=9, mfc="none",
                    mec="white", mew=2.0)
            ax.text(g_lams.index(l), g_ds.index(g) + 0.25, lab, fontsize=7.5,
                    color="white", ha="center")
    fig.colorbar(im, ax=ax, label="energy created [mJ]")
    ax.set_title("No cell is both passive and finishing\n"
                 "(one knob at a time never showed this)", fontsize=10)

    ax = axes[2]
    ax.scatter(dv, ev, s=26, color="0.45", label="grid cells (two gains)")
    ax.plot([po["depth"]], [max(po["e_ctrl"] * 1e3, 1e-2)], "*", ms=18,
            color="crimson", zorder=5, label="#66 observer (passive)")
    ax.axvline(DEPTH_BAR, color="0.3", ls=":", lw=1)
    ax.text(DEPTH_BAR + 0.4, ev.max() * 0.6, "task bar", fontsize=7.5, color="0.3")
    ax.annotate("off the curve entirely",
                xy=(po["depth"], max(po["e_ctrl"] * 1e3, 1e-2)),
                xytext=(po["depth"] - 18.0, 3.0), fontsize=8, color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    ax.set_yscale("symlog", linthresh=1e-1)
    ax.set_xlabel("depth reached [mm]"); ax.set_ylabel("energy created [mJ]")
    ax.set_title(f"The leak is the task\n(rank correlation {rho:.2f} across the grid)",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("67. The leftover injection was not an implementation defect — "
                 "it is what penetrating costs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "67_the_leak_is_the_task.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/67_the_leak_is_the_task.png, assets/67_the_leak_is_the_task.png")

    return dict(A=(a_off, a_jit), B=B, C=C, G=G, E=E, F=F, rho=rho)


if __name__ == "__main__":
    main()
