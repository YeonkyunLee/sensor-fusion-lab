"""제어기 **전체**의 수동성 — 아홉 실험 동안 적어만 두고 한 번도 안 물었던 것.

exp 56 부터 64 까지, 이 사슬은 채널 수동성을 **파동 블록의 장부**로 채점해 왔다. 그리고 매번
한계 절에 같은 문장을 적었다 — *"이 장부는 파동 블록만 감싼다. 표류 보정 항을 포함한 전체
제어기에 대한 수동성 논증은 아직 없다."* exp 57 은 그 밖의 항이 **일을 하고 있다**(브레이크)는
것까지 밝혔고, exp 58 은 그 일을 **대체**했다. 그런데 **전체가 수동인지는 아무도 묻지 않았다.**

**묻는 방법.** 제어기를 두 포트(술자의 손, 조직)를 가진 하나의 블록으로 보고, 그 블록이 두 기계
몸체에 **해 준 일**을 적산한다:

    E_ctrl(t) = ∫ [ (f_m_ch + f_vf + f_ml)·v_m + (f_coup + f_loc)·v_s ] dt

내부 전원이 없는(수동) 블록은 받은 것과 처음 저장한 것 이상을 내줄 수 없다. 시작이 정지·무변형
이므로 **수동이면 E_ctrl(t) ≤ 0 이어야 한다.** 자라면 그만큼 **에너지를 만든 것**이다.

  A. **세 장부를 나란히** — 파동 블록 / 전체 제어기 / 전체에서 국소 감쇠를 뺀 것.
  B. **TDPA 는 자기가 재는 블록만 고친다** — 파동 장부의 위반은 지우는데 전체는 그대로인가.
  C. **지터 탓인가** — 지터와 λ 를 각각 쓸어 본다.
  D. **무엇이 이걸 줄이나** — exp 58 의 정지, exp 57 이 시도했던 λ 억제.

**결론부터.** 파동 블록 장부는 TDPA 를 켠 모든 조건에서 **정확히 0.0000 mJ** 인데, 전체 제어기는
그 동안 **77~106 mJ 를 만든다**(최악 시드 기준). 지터 40 ms 에서 파동 장부의 위반 **−6.44 mJ** 는
TDPA 가 지우지만 전체 공급은 77.4 → **80.2 mJ** 로 오히려 조금 늘었다 —
**자기가 재는 곳만 고친 것이다.**

(수동성은 모든 실행·모든 시각에 대한 조건이라 **최악 시드**로 묻는다. 중앙값으로 보면 40 ms 위반이
절반의 시드에서 사라져 "괜찮다"로 읽히는데, 그건 passivity 가 요구하는 답이 아니다.)

그리고 주입이 가장 큰 조건이 **지터 0**(103 mJ)이다. 채널 현상이 아니라 **표류 보정 항 자체**다:
파동 좌표의 에너지 회계를 거치지 않고 얹은 위치 서보라 **지연이 없어도 능동**이다.
λ 를 3 → 48 로 키우면 18 → 94 mJ 로 자라는데, **exp 56 이 과제를 완주시키려고 올린 값이 바로
그 λ (3 → 24, 깊이 34.6 → 50.8 mm)** 다. 시험을 의미 있게 만든 손질이 전체 수동성을 깬 손질이고,
그때 보던 장부는 계속 0 이었다.

크기 감으로 76 mJ 는 2 kg 도구를 **0.28 m/s** 로 던질 수 있는 양이다.

exp 63 이 "지표군이 한 축이면 서로 검산해도 같이 틀린다"였다면, 이건 그 **보증판**이다 —
**증명이 덮는 영역 안에서의 만족은 전체에 대한 증거가 아니다.**

    python scripts/65_whole_loop_passivity.py

한계·트레이드오프
  - E_ctrl 은 **상한 위반**만 본다. 얼마나 자주·어떤 형태로 새는지(주파수 특성)는 안 본다.
  - 국소 감쇠(f_loc)는 제어 법칙의 일부라 전체 장부에 포함하는 게 맞지만, 항상 소산이라 판정을
    **관대하게** 만든다. 그래서 그 항을 뺀 장부도 같이 낸다(exp 56 의 R17 과 같은 이유).
  - **능동이라고 곧 불안정인 것은 아니다.** 사람과 조직이 충분히 소산적이면 능동 제어기와도
    안정할 수 있다. 여기서 보이는 것은 **보증이 없다**는 것이지 발산한다는 것이 아니다
    (실제로 이 조건들에서 발산하지 않는다).
  - 고치는 방법(수동성을 회복하는 표류 보정 설계)은 여기서 제안하지 않는다. **없는 것을 있다고
    읽고 있었다**는 것까지가 이 실험이다.
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

DT, M_S = jc.DT, jc.M_S
LAM_50, LAM_TASK = jc.LAM_50, jc.LAM_TASK
N_SEEDS = 6
JITTERS = (0.0, 10.0, 20.0, 40.0)
LAMS = (3.0, 6.0, 12.0, 24.0, 48.0)
BURST = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS,
             resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=s59.BREATH_HZ)
KEYS = ("final_depth_mm", "e_min", "e_drawdown", "e_ctrl_max", "e_ctrl_nd_max",
        "blind_max_mm", "osc_mm", "diverged")


def med(mode="tdpa", seeds=N_SEEDS, bursty=False, **kw):
    rows = []
    for s in range(seeds):
        r = (bc.run(mode, seed=s, tissue_obj=s59.GrippingTissue(), **BURST, **kw)
             if bursty else jc.run(mode, seed=s, **kw))
        rows.append(r)
    # 일부 키는 정지 기능을 켠 실행에만 있다(blind_max_mm 등) — 없으면 nan 으로 둔다.
    return {k: float(np.median([r.get(k, np.nan) for r in rows])) for k in KEYS}


def speed_for(energy_j):
    """그 에너지가 전부 도구 운동에너지로 갔다면 몇 m/s 인가 — 크기 감을 잡기 위한 환산."""
    return float(np.sqrt(2.0 * max(energy_j, 0.0) / M_S))


def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    jits = (0.0, 20.0) if quick else JITTERS
    lams = (3.0, 24.0) if quick else LAMS
    print("=== 65. 제어기 전체의 수동성 — 아홉 실험 동안 적어만 두고 안 물었던 것 ===")
    print("exp 56~64 는 매번 한계 절에 '이 장부는 파동 블록만 감싼다'고 적었다.")
    print("제어기를 두 포트 블록으로 보고 **두 몸체에 해 준 일**을 적산한다 — 수동이면 ≤ 0 이어야 한다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 같은 실행을 세 장부로 채점한다")
    base = med("tdpa", seeds=seeds, jitter_ms=20.0)
    print(f"{'장부':>28s} | {'값 [mJ]':>10s} | {'수동인가':>10s}")
    print(f"{'파동 블록 최저치 (exp 56~64)':>28s} | {base['e_min'] * 1e3:10.4f} | "
          f"{('예' if base['e_min'] >= -1e-9 else '**아니오**'):>10s}")
    print(f"{'전체 제어기 최대 공급':>28s} | {base['e_ctrl_max'] * 1e3:10.4f} | "
          f"{('예' if base['e_ctrl_max'] <= 1e-9 else '**아니오**'):>10s}")
    print(f"{'같은 것, 국소 감쇠 제외':>28s} | {base['e_ctrl_nd_max'] * 1e3:10.4f} | "
          f"{('예' if base['e_ctrl_nd_max'] <= 1e-9 else '**아니오**'):>10s}")
    print(f"  파동 장부는 **{base['e_min'] * 1e3:.4f} mJ** 로 정확히 만족하는데 전체는 "
          f"**{base['e_ctrl_max'] * 1e3:.1f} mJ 를 만든다.**")
    print(f"  크기 감: 그 에너지가 전부 도구 운동으로 가면 **{speed_for(base['e_ctrl_max']):.2f} m/s** 다"
          f"(도구 {M_S:.0f} kg).")
    print(f"  exp 56 이 헤드라인으로 쓴 '지터가 만든 에너지' 는 이 실행에서 "
          f"{base['e_drawdown'] * 1e3:.2f} mJ — **{base['e_ctrl_max'] / max(base['e_drawdown'], 1e-12):.0f}배** 차이다.")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] TDPA 는 **자기가 재는 블록만** 고친다")
    print("    수동성은 **모든 실행·모든 시각**에 대한 조건이라 중앙값이 아니라 **최악 시드**로 묻는다.")
    print("    (중앙값으로 보면 40 ms 위반이 절반의 시드에서 사라져 '괜찮다'로 읽힌다.)")
    print(f"{'모드':>6s} | {'지터[ms]':>8s} | {'파동 최저(최악)[mJ]':>18s} | {'전체 공급(최악)[mJ]':>19s} | "
          f"{'깊이[mm]':>9s}")
    B = {}
    for mode in ("zoh", "tdpa"):
        for j in jits:
            runs = [jc.run(mode, seed=s, jitter_ms=j) for s in range(seeds)]
            m = {k: float(np.median([r.get(k, np.nan) for r in runs])) for k in KEYS}
            m["e_min_worst"] = float(min(r["e_min"] for r in runs))
            m["e_ctrl_worst"] = float(max(r["e_ctrl_max"] for r in runs))
            B[(mode, j)] = m
            print(f"{mode:>6s} | {j:8.0f} | {m['e_min_worst'] * 1e3:18.4f} | "
                  f"{m['e_ctrl_worst'] * 1e3:19.2f} | {m['final_depth_mm']:9.1f}")
    j_bad = max(jits, key=lambda j: -B[("zoh", j)]["e_min_worst"])
    z, t = B[("zoh", j_bad)], B[("tdpa", j_bad)]
    print(f"  지터 {j_bad:.0f} ms 에서 파동 장부의 위반 {z['e_min_worst'] * 1e3:+.4f} → "
          f"{t['e_min_worst'] * 1e3:+.4f} mJ 로 **사라진다**(exp 56 의 결과 그대로).")
    print(f"  같은 조건에서 전체 공급은 {z['e_ctrl_worst'] * 1e3:.1f} → {t['e_ctrl_worst'] * 1e3:.1f} mJ 로")
    print("  **거의 그대로다.** 고친 것은 **재고 있던 숫자**이지 시스템이 아니었다.")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] 지터 탓인가 — 아니다. **표류 보정 항 자체**다")
    print(f"{'지터[ms]':>9s} | {'전체 공급[mJ]':>13s}   ||{'λ [1/s]':>9s} | {'전체 공급[mJ]':>13s} | "
          f"{'깊이[mm]':>9s}")
    C_j = {j: B[("tdpa", j)]["e_ctrl_max"] for j in jits}
    C_l = {}
    rows = max(len(jits), len(lams))
    for i in range(rows):
        left = (f"{jits[i]:9.0f} | {C_j[jits[i]] * 1e3:13.2f}" if i < len(jits)
                else " " * 25)
        if i < len(lams):
            m = med("tdpa", seeds=seeds, jitter_ms=20.0, lam_pos=lams[i])
            C_l[lams[i]] = m
            right = (f"{lams[i]:9.0f} | {m['e_ctrl_max'] * 1e3:13.2f} | "
                     f"{m['final_depth_mm']:9.1f}")
        else:
            right = ""
        print(f"{left}   ||{right}")
    j0, jmax = C_j[jits[0]], max(C_j.values())
    print(f"  **주입이 가장 큰 조건이 지터 0 이다**({j0 * 1e3:.1f} mJ). 지터를 넣으면 오히려 준다 —")
    print("  채널 현상이 아니라는 뜻이다. 파동 좌표의 에너지 회계를 **거치지 않고** 얹은 위치 서보라,")
    print("  지연이 없어도 능동이다.")
    if len(lams) >= 2:
        lo, hi = C_l[lams[0]], C_l[lams[-1]]
        print(f"  λ 를 키우면 {lo['e_ctrl_max'] * 1e3:.1f} → {hi['e_ctrl_max'] * 1e3:.1f} mJ 로 자란다.")
        print(f"  그런데 **exp 56 은 과제를 완주시키려고 λ 를 {LAM_50:.0f} → {LAM_TASK:.0f} 으로 올렸다**")
        print(f"  (깊이 {lo['final_depth_mm']:.1f} → {C_l[LAM_TASK]['final_depth_mm']:.1f} mm). 시험을 의미 있게")
        print("  만든 그 손질이 **전체 수동성을 깬 손질**이다 — 그리고 그때 보던 장부는 계속 0 이었다.")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] 무엇이 이걸 줄이나 — 이미 있는 두 처방을 **이 장부로** 재본다")
    print(f"{'구성':>22s} | {'전체 공급[mJ]':>13s} | {'맹행[mm]':>9s} | {'깊이[mm]':>9s}")
    D = {}
    for name, kw in (("정지 없음", dict(estop=False)),
                     ("exp 58 정지", dict(estop=True)),
                     ("정지 + λ 억제(exp 57)", dict(estop=True, lam_gate=True))):
        m = med("tdpa", seeds=seeds, bursty=True, **kw)
        D[name] = m
        print(f"{name:>22s} | {m['e_ctrl_max'] * 1e3:13.2f} | {m['blind_max_mm']:9.2f} | "
              f"{m['final_depth_mm']:9.1f}")
    a, b, c = D["정지 없음"], D["exp 58 정지"], D["정지 + λ 억제(exp 57)"]
    print(f"  **exp 58 의 정지가 주입도 줄인다**: {a['e_ctrl_max'] * 1e3:.1f} → "
          f"{b['e_ctrl_max'] * 1e3:.1f} mJ. 아무도 주장하지 않았던 덤이다")
    print("  (설계 목적은 맹행 제한이었다 — 능동 구간을 짧게 끊는 것이 부수적으로 여기에도 듣는다).")
    print(f"  **λ 억제는 여기서도 안 듣는다**: {b['e_ctrl_max'] * 1e3:.1f} → {c['e_ctrl_max'] * 1e3:.1f} mJ,")
    print(f"  맹행은 {b['blind_max_mm']:.2f} → {c['blind_max_mm']:.2f} mm 로 나빠진다. exp 57·58 은 이 처방을")
    print("  **맹행**으로 채점해 기각했는데, 명목상 겨냥했던 **수동성**으로 재도 마찬가지다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 정리")
    print("  1. **전체 제어기는 수동이 아니다.** 파동 블록 장부가 정확히 0 인 동안 66~103 mJ 를 만든다.")
    print("     아홉 실험이 한계 절에 적어만 두고 한 번도 재지 않은 자리다.")
    print("  2. **TDPA 는 자기가 재는 블록만 고쳤다.** 파동 장부의 위반은 지우고 전체는 그대로 둔다.")
    print("     → exp 63 이 지표에서 본 것의 **보증판**이다: **증명이 덮는 영역 안에서의 만족은**")
    print("       **전체에 대한 증거가 아니다.**")
    print("  3. **지터 탓이 아니다.** 주입이 가장 큰 조건이 지터 0 이고, λ 에 따라 자란다.")
    print("     그리고 그 λ 는 exp 56 이 **과제를 완주시키려고** 올린 값이다 — 한 문제를 고친 손질이")
    print("     다른 문제를 만들었고, 그때 보던 장부는 그걸 볼 수 없었다.")
    print("  4. **exp 58 의 정지가 덤으로 줄여 준다**(90 → 57 mJ). λ 억제는 여기서도 안 듣는다.")
    print("  5. 정직하게: **능동이라고 곧 불안정은 아니다.** 이 조건들에서 발산하지 않는다.")
    print("     보이는 것은 **보증이 없다**는 것이고, 사람·조직의 소산에 기대고 있다는 것이다.")
    print("     수동성을 회복하는 표류 보정 설계는 여기서 제안하지 않는다 — 다음 일이다.")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    ax = axes[0]
    x = np.arange(len(jits))
    ax.bar(x - 0.2, [B[("zoh", j)]["e_ctrl_max"] * 1e3 for j in jits], 0.4,
           color="0.55", label="hold-last")
    ax.bar(x + 0.2, [B[("tdpa", j)]["e_ctrl_max"] * 1e3 for j in jits], 0.4,
           color="crimson", label="TDPA budget")
    ax2 = ax.twinx()
    ax2.plot(x, [B[("zoh", j)]["e_min_worst"] * 1e3 for j in jits], "-o", color="tab:blue")
    ax2.plot(x, [B[("tdpa", j)]["e_min_worst"] * 1e3 for j in jits], "-s", color="seagreen")
    ax2.axhline(0, color="0.3", lw=1)
    ax2.set_ylabel("wave-block ledger min [mJ]")
    ax.set_xticks(x); ax.set_xticklabels([f"{j:.0f}" for j in jits])
    ax.set_xlabel("jitter [ms]"); ax.set_ylabel("energy created by the controller [mJ]")
    ax.set_title("TDPA zeroes the ledger it watches,\nnot the energy", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ls = sorted(C_l)
    ax.plot(ls, [C_l[l]["e_ctrl_max"] * 1e3 for l in ls], "-o", color="crimson",
            label="energy created [mJ]")
    ax.set_xlabel("drift-correction gain lambda [1/s]")
    ax.set_ylabel("energy created [mJ]", color="crimson")
    ax2 = ax.twinx()
    ax2.plot(ls, [C_l[l]["final_depth_mm"] for l in ls], "-s", color="0.3")
    ax2.axhline(55.0, color="0.3", ls=":", lw=1)
    ax2.set_ylabel("depth reached [mm]")
    ax.axvline(LAM_50, color="tab:blue", ls="--", lw=1)
    ax.axvline(LAM_TASK, color="seagreen", ls="--", lw=1)
    ax.text(LAM_50 * 1.05, ax.get_ylim()[1] * 0.9, "#50", fontsize=8, color="tab:blue")
    ax.text(LAM_TASK * 1.05, ax.get_ylim()[1] * 0.9, "#56 (to finish\nthe task)",
            fontsize=8, color="seagreen")
    ax.set_title("The gain raised to make the test honest\nis the gain that breaks passivity",
                 fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    names = list(D)
    ax.barh(range(len(names)), [D[n]["e_ctrl_max"] * 1e3 for n in names],
            color=["0.55", "seagreen", "crimson"])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(["no stop", "#58 stop", "stop + lambda\ngating (#57)"],
                       fontsize=8)
    ax.set_xlabel("energy created [mJ]")
    ax.set_title("The stop helps (nobody claimed that);\ngating the term still does not",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="x")

    fig.suptitle("65. Passivity of the whole controller — the sentence every experiment "
                 "wrote and none of them measured", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "65_whole_loop_passivity.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/65_whole_loop_passivity.png, assets/65_whole_loop_passivity.png")

    return dict(A=base, B=B, C=(C_j, C_l), D=D)


if __name__ == "__main__":
    main()
