"""아무도 안 건드린 세 번째 축 — 그리고 **완주 기준이 바닥이면 초과를 허가한다**.

exp 67 은 (결합 이득, 표류 이득) 격자를 채우고 *"수동이면서 완주하는 칸은 없다"* 로 끝났다.
**그건 2 차원 절단면 위에서의 진술이었다.** 파동 임피던스 b 자체는 exp 50 이 고른 10 에
**열여덟 실험 동안 고정**돼 있었고, 아무도 그 축을 움직이지 않았다. 움직여 보면 칸이 있다.

  A. **먼저 선불(prepay)을 시도한다** — 수동성은 E_ctrl(t) ≤ 0 이니 접근 구간에 미리 음수로
     쌓아 두고 침투 구간에 쓰면 되지 않나. (이게 이 실험의 첫 설계였고, 실패했다.)
  B. **세 번째 축** — b 를 올리면 무엇이 달라지는가. exp 67 은 이득을 **내려서** 비율을
     맞췄고(과제 상실), 이쪽은 임피던스를 **올려서** 맞춘다(과제 유지).
  C. **3 차원 격자** — '수동이면서 완주'가 존재하는가.
  D. **그 칸이 무엇을 하는가** — 도달 깊이를 표적과 나란히 놓고 본다.
  E. **대가** — 힘 투명성·떨림·**술자의 손이 하는 일**.
  F. **최악 조건** — 사슬이 도달한 최악에서도 버티는가.

**미리 밝히는 결론.** 세 가지를 예측했고 **하나가 틀렸고 하나는 절반만 맞았다.**

  1. "선불로 미리 쌓아 두면 침투를 감당한다" — **틀렸다.** 자연 적립이 **−0.017 mJ** 뿐인데
     필요한 것은 80 mJ 이고, 일부러 쌓으면 주입이 **오히려 는다**(80.7 → 101.7 mJ). 자유공간에서
     팔을 붙잡으면 손과 도구가 어긋나는데, **그 어긋남이 바로 주입이 지불하는 대상**이다.
     선불이 자기를 먹는다.
  2. "임피던스를 올리면 격자 밖으로 나간다" — **맞았다, 깨끗한 채널에서만.** b = 480(exp 50
     값의 48 배)에서 **12 시드 전부 위반 0** 이면서 완주하는 칸이 여덟 개 있다. 그런데
     **최악 조건에서는 6/12 가 위반**하고 끌림이 51% 나빠진다.
  3. "임피던스를 올리면 힘 투명성을 판다" — **맞았다.** 힘 오차가 0.37 → 2.57 N(7 배)이고,
     덤으로 **술자의 손이 하는 일이 3.3 배**가 된다(34.7 → 113.9 mJ). 파동 변환이 장식이 되고
     팔이 사실상 위치 서보가 되는 값이다.

그리고 이 실험의 진짜 소득은 설계가 아니라 **기준의 결함**이다. 그 수동 칸은 도구를
**59.1 mm** 까지 밀어 넣는다 — 표적은 **55 mm** 다. 그런데 사슬의 완주 기준 R18 은
*"45 mm 이상 도달"* 이라 **바닥이지 창이 아니다.** 즉 이 칸은 **사슬이 가진 모든 검사를
통과하면서** 표적을 4 mm 지나친다. 열여덟 실험이 초과를 볼 수 없는 자로 재고 있었다.

    python scripts/68_the_axis_nobody_moved.py

한계·트레이드오프
  - **격자는 여전히 격자다.** 4×3×3 을 3 시드로 봤다(C 절). exp 67 이 2 차원에서 놓친 것을
    3 차원에서 찾았다는 사실 자체가 **다음 축이 또 있을 수 있다**는 뜻이기도 하다 — 여기서
    확정할 수 있는 것은 "b 를 고정한 결론은 b 에 대한 결론이 아니다"까지다.
  - **b = 480 의 수동성은 지터 채널의 성질**이다(F 절에서 6/12 로 무너진다). exp 66 이
    시드 수로 배운 것과 같은 종류의 조건부이고, 여기서는 **채널 조건**이 그 역할을 한다.
  - **표적 초과를 위해로 환산하지 않았다.** 59.1 mm 가 55 mm 표적에서 임상적으로 무엇을
    뜻하는지는 이 사슬 밖의 숫자다(exp 63 의 끌림 축에서는 최악 조건에서 51% 나빠지는 것까지만
    말할 수 있다). 다만 **기준이 그걸 볼 수 없다**는 것은 사슬 안의 결함이고, 그건 고칠 수 있다.
  - **선불은 한 가지 형태만 시험했다** — 자유공간 감쇠. 마스터 쪽에서 걷거나 침투 중에 갚는
    형태는 안 봤다. "선불이 불가능하다"가 아니라 **"이 선불은 자기를 먹는다"** 까지다.
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
tele = import_module("50_teleoperation_delay")
s63 = import_module("63_harm_is_not_force")

B_CHAIN = jc.B_WAVE           # exp 50 이 고르고 열여덟 실험이 안 건드린 값
D_S_CHAIN = jc.D_S
LAM_CHAIN = jc.LAM_TASK
TARGET_MM = tele.X_TARGET * 1e3   # 표적 깊이 — R18 이 못 보는 쪽
DEPTH_BAR = 45.0                  # R18(exp 56) — **바닥**이다
JITTER = 20.0
N_SEEDS = 6
COST_SEEDS = 12               # 수동성 주장은 exp 66 이 배운 대로 시드를 늘려서
GRID_SEEDS = 3
HARM_SEEDS = s63.N_SEEDS

PREPAYS = (0.0, 50.0, 200.0, 800.0, 3000.0)
B_LADDER = (10.0, 20.0, 40.0, 60.0, 120.0)
GRID_B = (10.0, 60.0, 240.0, 480.0)
GRID_DS = (30.0, 60.0, 120.0)
GRID_LAMS = (12.0, 24.0, 48.0)
PASSIVE_CELL = dict(b_wave=480.0, d_s=60.0, lam_pos=24.0)


def clean(seeds=N_SEEDS, **kw):
    """지터만 있는 채널. 수동성은 **최악**, 나머지는 중앙값 — exp 65 이후의 규칙."""
    rs = [jc.run("tdpa", seed=s, jitter_ms=JITTER, **kw) for s in range(seeds)]
    ok = [r for r in rs if np.isfinite(r["final_depth_mm"])]
    if not ok:
        return dict(e_ctrl=np.inf, bad=list(range(len(rs))), depth=np.nan, dmax=np.nan,
                    osc=np.inf, ferr=np.inf, perr=np.inf, hand=np.nan, reserve=np.nan)
    ev = [r["e_ctrl_max"] for r in ok]
    return dict(
        e_ctrl=max(ev),
        bad=[i for i, v in enumerate(ev) if v > 1e-9],
        depth=float(np.median([r["final_depth_mm"] for r in ok])),
        dmax=max(r["final_depth_mm"] for r in ok),
        osc=float(np.median([r["osc_mm"] for r in ok])),
        ferr=float(np.median([r["force_err_N"] for r in ok])),
        perr=float(np.median([r["pos_err_mm"] for r in ok])),
        hand=float(np.median([r["e_hand"] for r in ok])),
        # **적립 가능한 여유** — 장부가 얼마나 음수로 내려가는가. 선불 설계의 재원이다.
        reserve=float(np.median([min(r["log"]["e_ctrl"]) for r in ok])),
    )


def harsh(seeds=HARM_SEEDS, **kw):
    """exp 57 연집 + exp 58 정지 + exp 63 끌림 조직 — 사슬이 도달한 최악."""
    rs = [bc.run("tdpa", seed=s,
                 tissue_obj=s63.DraggingTissue(tau=s63.TAU, f_slip=s63.F_SLIP0),
                 **s63.BASE, **kw) for s in range(seeds)]
    ev = [r["e_ctrl_max"] for r in rs]
    return dict(e_ctrl=max(ev), bad=[i for i, v in enumerate(ev) if v > 1e-9],
                depth=float(np.median([r["final_depth_mm"] for r in rs])),
                drag=float(np.median([r["drag_total_mm"] for r in rs])),
                drag_held=float(np.median([r["drag_held_mm"] for r in rs])))


def scan3(bs=GRID_B, ds=GRID_DS, lams=GRID_LAMS, seeds=GRID_SEEDS):
    """3 차원 격자. **exp 67 이 채운 것은 이 상자의 b = 10 면 하나였다.**"""
    out = {}
    for b in bs:
        for g in ds:
            for lam in lams:
                rs = [jc.run("tdpa", seed=s, jitter_ms=JITTER,
                             b_wave=b, d_s=g, lam_pos=lam) for s in range(seeds)]
                ok = [r for r in rs if np.isfinite(r["final_depth_mm"])]
                if not ok:
                    continue
                out[(b, g, lam)] = dict(
                    e_ctrl=max(r["e_ctrl_max"] for r in ok),
                    depth=float(np.median([r["final_depth_mm"] for r in ok])))
    return out


def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    cseeds = 3 if quick else COST_SEEDS
    gseeds = 1 if quick else GRID_SEEDS
    preps = (0.0, 3000.0) if quick else PREPAYS
    blad = (10.0, 120.0) if quick else B_LADDER
    gb = (10.0, 480.0) if quick else GRID_B
    gd = (60.0,) if quick else GRID_DS
    gl = (24.0,) if quick else GRID_LAMS
    hseeds = 3 if quick else HARM_SEEDS

    print("=== 68. 아무도 안 건드린 세 번째 축 — 그리고 완주 기준이 바닥이면 초과를 허가한다 ===")
    print(f"exp 50 이 파동 임피던스를 b = {B_CHAIN:.0f} 으로 골랐고, 이후 열여덟 실험이 "
          f"그 값을 한 번도 안 움직였다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 먼저 **선불**을 시도한다 — 수동성은 E_ctrl(t) ≤ 0 이니 미리 음수로 쌓아 두면 된다")
    r0 = jc.run("tdpa", seed=0, jitter_ms=JITTER)
    e_tr = np.array(r0["log"]["e_ctrl"]) * 1e3
    t_tr = np.array(r0["log"]["t"])
    fe_tr = np.abs(np.array(r0["log"]["fe"]))
    touch = int(np.argmax(fe_tr > 1e-3)) if (fe_tr > 1e-3).any() else 0
    print(f"  먼저 **언제 새는지** 본다. 조직에 처음 닿는 시각은 t = {t_tr[touch]:.2f} s 인데,")
    print(f"  그 시점에 장부가 이미 **{e_tr[touch]:+.1f} mJ** 다(최종 {e_tr.max():.1f} mJ).")
    print(f"  **{e_tr[touch] / max(e_tr.max(), 1e-9) * 100:.0f}% 가 조직에 닿기 전에 쌓인다** — "
          "exp 67 의 '침투의 가격'은 침투 구간만의 얘기가 아니다.")
    print(f"  그리고 **쌓이는 여유는 없다**: 장부의 최솟값이 {e_tr.min():+.3f} mJ 다"
          f"(필요한 것은 {e_tr.max():.0f} mJ).")
    print()
    print(f"{'선불 감쇠':>9s} | {'E_ctrl[mJ]':>11s} | {'적립[mJ]':>10s} | {'깊이[mm]':>9s} | "
          f"{'위치오차[mm]':>12s} | {'손이 한 일[mJ]':>14s}")
    A = {}
    for d in preps:
        m = clean(seeds=seeds, prepay_d=d)
        A[d] = m
        print(f"{d:9.0f} | {m['e_ctrl'] * 1e3:11.3f} | {m['reserve'] * 1e3:10.2f} | "
              f"{m['depth']:9.2f} | {m['perr']:12.3f} | {m['hand'] * 1e3:14.1f}")
    a0, a1 = A[preps[0]], A[preps[-1]]
    print(f"  **선불이 자기를 먹는다.** 적립은 {a1['reserve'] * 1e3:.1f} mJ 에서 멈추는데 주입은 "
          f"{a0['e_ctrl'] * 1e3:.1f} → {a1['e_ctrl'] * 1e3:.1f} mJ 로 **늘었다.**")
    print(f"  자유공간에서 팔을 붙잡으면 손과 도구가 어긋나고(위치 오차 {a0['perr']:.1f} → "
          f"{a1['perr']:.1f} mm),")
    print("  **그 어긋남이 바로 표류 항이 지불하는 대상**이다. 더 세게 걸면 발산한다.")
    print("  → **예측 1 이 틀렸다.** 미리 갚는 설계는 갚을 재원을 만들면서 빚을 같이 만든다.")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] 그럼 **세 번째 축** — exp 67 은 이득을 내려 비율을 맞췄다(과제 상실). 반대로 간다")
    print(f"    (exp 67: d_s 를 {B_CHAIN:.0f} 까지 내리면 수동, 대신 27.5 mm. 여기서는 b 를 올린다.)")
    print(f"{'b':>7s} | {'d_s/b':>6s} | {'E_ctrl[mJ]':>11s} | {'깊이[mm]':>9s} | {'진동[mm]':>9s} | "
          f"{'힘오차[N]':>10s} | {'손[mJ]':>9s}")
    B = {}
    for b in blad:
        m = clean(seeds=seeds, b_wave=b)
        B[b] = m
        print(f"{b:7.0f} | {D_S_CHAIN / b:6.1f} | {m['e_ctrl'] * 1e3:11.3f} | {m['depth']:9.2f} | "
              f"{m['osc']:9.3f} | {m['ferr']:10.3f} | {m['hand'] * 1e3:9.1f}")
    b0, b1 = B[blad[0]], B[blad[-1]]
    print(f"  **같은 깊이에서 주입이 {b0['e_ctrl'] * 1e3:.1f} → {b1['e_ctrl'] * 1e3:.1f} mJ**"
          f"({(1 - b1['e_ctrl'] / b0['e_ctrl']) * 100:.0f}% 감소) 다.")
    print(f"  대가는 예측대로 **투명성**이다: 힘 오차 {b0['ferr']:.2f} → {b1['ferr']:.2f} N, "
          f"진동 {b0['osc']:.2f} → {b1['osc']:.2f} mm.")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] **3 차원 격자** — exp 67 이 채운 것은 이 상자의 b = 10 면 하나였다")
    G = scan3(bs=gb, ds=gd, lams=gl, seeds=gseeds)
    fin = {k: v for k, v in G.items() if v["depth"] >= DEPTH_BAR}
    both = {k: v for k, v in fin.items() if v["e_ctrl"] <= 1e-9}
    print(f"{'b':>7s} | {'d_s':>6s} | {'λ':>5s} | {'E_ctrl[mJ]':>11s} | {'깊이[mm]':>9s} | "
          f"{'수동+완주?':>10s}")
    for (b, g, lam), v in sorted(G.items()):
        flag = "**예**" if (v["e_ctrl"] <= 1e-9 and v["depth"] >= DEPTH_BAR) else ""
        print(f"{b:7.0f} | {g:6.0f} | {lam:5.0f} | {v['e_ctrl'] * 1e3:11.4f} | "
              f"{v['depth']:9.2f} | {flag:>10s}")
    slice10 = {k: v for k, v in G.items() if k[0] == B_CHAIN}
    print(f"  **b = {B_CHAIN:.0f} 면에서는 {sum(1 for v in slice10.values() if v['e_ctrl'] <= 1e-9 and v['depth'] >= DEPTH_BAR)} 칸** — exp 67 이 본 그대로다.")
    print(f"  **상자 전체에서는 {len(both)} 칸.** exp 67 의 '그런 칸은 없다'는 "
          "**2 차원 절단면 위에서의 진술**이었다.")
    print("  → **한 축을 고정한 채 얻은 결론은 그 축에 대한 결론이 아니다.** 그리고 그 축은")
    print(f"     exp 50 이 고른 뒤 **열여덟 실험 동안 아무도 안 움직였다.**")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] 그런데 **그 칸이 무엇을 하는가** — 도달 깊이를 **표적과 나란히** 놓고 본다")
    cell = clean(seeds=cseeds, **PASSIVE_CELL)
    base = clean(seeds=cseeds)
    obs = clean(seeds=cseeds, drift_mode="po", po_strict=True)
    print(f"{'구성':>22s} | {'E_ctrl[mJ]':>11s} | {'위반':>7s} | {'깊이[mm]':>9s} | "
          f"{'최대깊이':>9s} | {'표적 대비':>10s}")
    D = {"사슬": base, f"b={PASSIVE_CELL['b_wave']:.0f} 수동 칸": cell, "exp 66 관측기": obs}
    for name, m in D.items():
        viol = "%d/%d" % (len(m["bad"]), cseeds)
        print(f"{name:>22s} | {m['e_ctrl'] * 1e3:11.4f} | {viol:>7s} | {m['depth']:9.2f} | "
              f"{m['dmax']:9.2f} | {m['depth'] - TARGET_MM:+10.2f}")
    print(f"  표적은 **{TARGET_MM:.0f} mm** 인데 그 수동 칸은 **{cell['depth']:.1f} mm** 까지 민다 "
          f"({cell['depth'] - TARGET_MM:+.1f} mm).")
    print(f"  그런데 사슬의 완주 기준 **R18 은 '{DEPTH_BAR:.0f} mm 이상'** 이다 — **바닥이지 창이 아니다.**")
    print("  **이 칸은 사슬이 가진 모든 검사를 통과하면서 표적을 지나친다.**")
    print("  열여덟 실험이 초과를 **볼 수 없는 자**로 재고 있었다. 이 실험의 진짜 소득은 설계가")
    print("  아니라 **기준의 결함**이다 — 완주를 바닥으로 쓰면 그 바닥을 넘는 나쁜 설계가 통과한다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] **대가** — 힘 투명성·떨림, 그리고 **청구서가 누구에게 가는가**")
    print(f"{'구성':>22s} | {'진동[mm]':>9s} | {'힘오차[N]':>10s} | {'위치오차[mm]':>12s} | "
          f"{'손이 한 일[mJ]':>14s}")
    for name, m in D.items():
        print(f"{name:>22s} | {m['osc']:9.3f} | {m['ferr']:10.3f} | {m['perr']:12.3f} | "
              f"{m['hand'] * 1e3:14.1f}")
    print(f"  힘 오차가 {base['ferr']:.2f} → {cell['ferr']:.2f} N "
          f"({cell['ferr'] / max(base['ferr'], 1e-9):.0f} 배)다 — **술자가 조직을 거의 못 느낀다.**")
    print(f"  그리고 **술자의 손이 하는 일이 {base['hand'] * 1e3:.1f} → {cell['hand'] * 1e3:.1f} mJ"
          f"({cell['hand'] / max(base['hand'], 1e-9):.1f} 배)** 다. 보증의 청구서가 사람에게 간다.")
    print("  임피던스를 48 배로 올린다는 것은 팔을 사실상 위치 서보로 만드는 것이라,")
    print("  **파동 변환이 장식이 된다.** 수동성은 그 대가로 온 것이다.")
    print(f"  (관측기는 같은 자리에서 힘 오차 {obs['ferr']:.2f} N 을 지킨다 — 대신 위치 오차가 "
          f"{obs['perr']:.1f} mm 다. **두 처방이 서로 다른 축을 판다.**)")

    # ------------------------------ F ------------------------------
    print("-" * 100)
    print("[F] **최악 조건** — 연집 손실 + 정지 + 끌림 조직에서도 버티나")
    print(f"{'구성':>22s} | {'E_ctrl[mJ]':>11s} | {'위반':>8s} | {'깊이[mm]':>9s} | "
          f"{'끌림[mm]':>9s} | {'정지중':>8s}")
    F = {}
    for name, kw in (("사슬", {}), (f"b={PASSIVE_CELL['b_wave']:.0f} 수동 칸", PASSIVE_CELL),
                     ("exp 66 관측기", dict(drift_mode="po", po_strict=True))):
        m = harsh(seeds=hseeds, **kw)
        F[name] = m
        viol = "%d/%d" % (len(m["bad"]), hseeds)
        print(f"{name:>22s} | {m['e_ctrl'] * 1e3:11.4f} | {viol:>8s} | {m['depth']:9.2f} | "
              f"{m['drag']:9.2f} | {m['drag_held']:8.2f}")
    hb, hc, ho = F["사슬"], F[f"b={PASSIVE_CELL['b_wave']:.0f} 수동 칸"], F["exp 66 관측기"]
    print(f"  **수동 칸이 무너진다** — {len(hc['bad'])}/{hseeds} 시드가 위반하고 "
          f"{hc['e_ctrl'] * 1e3:.1f} mJ 로 사슬({hb['e_ctrl'] * 1e3:.1f})과 거의 같아진다.")
    print(f"  그리고 **위해 축에서도 진다**: 끌림이 {hb['drag']:.1f} → {hc['drag']:.1f} mm 로 "
          f"{(hc['drag'] / hb['drag'] - 1) * 100:.0f}% 나빠진다.")
    print(f"  같은 조건에서 관측기는 {ho['e_ctrl'] * 1e3:.2f} mJ · {len(ho['bad'])}/{hseeds} · "
          f"끌림 {ho['drag']:.1f} mm 로 버틴다.")
    print("  → **예측 2 는 절반만 맞았다.** 칸은 있는데 **깨끗한 채널의 성질**이다.")

    # ------------------------------ G ------------------------------
    print("-" * 100)
    print("[G] 정리")
    print("  1. **exp 67 의 '그런 칸은 없다'는 2 차원 절단면의 결과였다.** 세 번째 축을 움직이면")
    print(f"     수동이면서 완주하는 칸이 있다 — 단 b = {PASSIVE_CELL['b_wave']:.0f}, "
          f"exp 50 값의 {PASSIVE_CELL['b_wave'] / B_CHAIN:.0f} 배에서.")
    print("     **한 축을 고정하고 얻은 결론은 그 축에 대한 결론이 아니다.**")
    print(f"  2. **그런데 그 칸은 표적을 지나친다**({cell['depth']:.1f} vs 표적 {TARGET_MM:.0f} mm)."
          " 그리고")
    print("     **R18 이 바닥이라 사슬의 어떤 검사도 그걸 못 본다.** 이게 이 실험의 진짜 소득이다 —")
    print("     **완주 기준을 바닥으로 쓰면 그 바닥을 넘는 나쁜 설계가 통과한다.**")
    print(f"  3. **대가는 사람에게 간다.** 힘 오차 {cell['ferr'] / max(base['ferr'], 1e-9):.0f} 배, "
          f"술자가 하는 일 {cell['hand'] / max(base['hand'], 1e-9):.1f} 배.")
    print("  4. **그리고 최악 조건에서 무너진다.** 그 수동성은 채널 조건부다.")
    print("  5. **선불은 자기를 먹는다.** 갚을 재원을 만드는 행위가 빚을 같이 만든다.")
    print("  ⇒ 세 번째 축이 준 것은 **새 설계가 아니라 exp 67 에 대한 정정과 기준의 결함**이다.")
    print("     exp 66 의 관측기는 여전히 최악 조건까지 가는 유일한 물건이다.")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    ax.plot(t_tr, e_tr, color="crimson", lw=1.4, label="two-port ledger")
    ax.axvline(t_tr[touch], color="0.3", ls=":", lw=1.2)
    ax.text(t_tr[touch] + 0.05, e_tr.max() * 0.55, "first tissue contact",
            fontsize=7.5, color="0.3")
    ax.axhline(0.0, color="0.4", lw=1.0)
    ax.annotate(f"{e_tr[touch]:.0f} mJ already spent\nbefore touching anything",
                xy=(t_tr[touch], e_tr[touch]), xytext=(t_tr[touch] + 0.4, e_tr.max() * 0.18),
                fontsize=8, arrowprops=dict(arrowstyle="->", color="0.3", lw=1.1))
    ax.set_xlabel("time [s]"); ax.set_ylabel("energy created [mJ]")
    ax.set_title("There is no reserve to save up\n"
                 f"(ledger minimum {e_tr.min():+.3f} mJ; the bill is {e_tr.max():.0f})",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    bb = list(blad)
    ax.plot(bb, [B[b]["e_ctrl"] * 1e3 for b in bb], "-o", color="crimson",
            label="energy created")
    ax.set_xscale("log"); ax.set_xlabel("wave impedance  b   (#50 chose 10)")
    ax.set_ylabel("energy created [mJ]", color="crimson")
    ax.axvline(B_CHAIN, color="0.3", ls=":", lw=1.2)
    ax.text(B_CHAIN * 1.1, B[bb[0]]["e_ctrl"] * 1e3 * 0.55,
            "held here for\n18 experiments", fontsize=7.5, color="0.3")
    ax2 = ax.twinx()
    ax2.plot(bb, [B[b]["ferr"] for b in bb], "-s", color="seagreen",
             label="force error (transparency)")
    ax2.set_ylabel("force error [N]", color="seagreen")
    ax.set_title("The third axis: injection falls, transparency is sold",
                 fontsize=10)
    ax.grid(alpha=0.3)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    ax = axes[2]
    names = ["chain", f"b={PASSIVE_CELL['b_wave']:.0f}\n(passive)", "#66\nobserver"]
    dep = [base["depth"], cell["depth"], obs["depth"]]
    x = np.arange(len(names))
    cols = ["0.55", "crimson", "steelblue"]
    ax.bar(x, dep, 0.55, color=cols)
    ax.axhline(TARGET_MM, color="darkorange", lw=2.0)
    ax.text(2.45, TARGET_MM + 0.5, "target", fontsize=8, color="darkorange", ha="right")
    ax.axhline(DEPTH_BAR, color="0.3", ls=":", lw=1.4)
    ax.text(2.45, DEPTH_BAR + 0.5, "R18 bar (a floor)", fontsize=8, color="0.3", ha="right")
    for i, v in enumerate(dep):
        ax.text(x[i], v + 0.4, f"{v:.1f}", ha="center", fontsize=8)
    ax.annotate("passes every check\nthe chain owns —\nand overshoots",
                xy=(1, cell["depth"]), xytext=(0.05, TARGET_MM + 6.0), fontsize=8,
                color="crimson",
                arrowprops=dict(arrowstyle="->", color="crimson", lw=1.2))
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("depth reached [mm]"); ax.set_ylim(0, max(dep) + 9)
    ax.set_title("A completion bar stated as a floor\nlicenses overshoot", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("68. The axis nobody moved — and a completion bar that cannot see overshoot",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "68_the_axis_nobody_moved.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/68_the_axis_nobody_moved.png, assets/68_the_axis_nobody_moved.png")

    return dict(A=A, B=B, G=G, D=D, F=F, both=both, target=TARGET_MM)


if __name__ == "__main__":
    main()
