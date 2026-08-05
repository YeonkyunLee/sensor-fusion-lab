"""원격조작 IV: 끊기면 멈추기 — exp 57 이 "미해결"로 남긴 것을 설계해 닫는다.

exp 57 의 결론 두 개가 서로 얽혀 있었다.
  · 연집 손실이 길어져도 도구가 멀리 안 가는 이유는 **낡은 명령을 붙들고 있는 것이 자기 제한적**이라
    팔이 정지 평형으로 수렴하기 때문이다.
  · 그 브레이크를 만드는 항이 exp 56 이 "수동성 증명이 덮지 않는 결함"으로 지목한 표류 보정
    λ(x_m − x_s) 였다. 그래서 그 항을 예산으로 함께 죄면 **오히려 나빠졌다**.

exp 57 은 거기서 "그러니 없애기 전에 그 항이 하는 일을 먼저 대체하라"는 규칙만 적고 멈췄다.
이 실험이 그 대체물을 **만들고 검증한다.** 순서는 세 단계다.

  A. **지금의 안전이 설계된 것인지 우연인지 절제로 가른다.** 조직 반력·표류 보정·국소 감쇠를
     하나씩 빼고 같은 암전에서 도구가 얼마나 가는지 본다.
     결과: 맹행이 **2.14 ~ 28.86 mm** 로 벌어진다 — 어느 항이 남아 있느냐에 달려 있다.
     즉 지금의 경계는 **설계가 아니라 이 플랜트가 우연히 갖고 있던 성질**이다.
  B. **통신 상실 정지를 설계해 넣는다.** 두 가지를 지킨다 —
       판정: **새 표본 없이 움직인 누적 거리**가 사슬이 이미 선언해 둔 여유를 넘을 때.
             통신 임계값이 아니라 **해부·계획에서 오는 숫자**다(exp 45 shaft 2.17 mm,
             exp 48 통로 1.25 mm 계열). 처음엔 "예산이 아무것도 허락하지 않는 순간(β=0)"을
             그대로 트리거로 썼는데, 지터가 방향별 80% 를 굶기는 채널에서는 그게 너무 자주
             성립해 **정지가 98.5% 걸려 과제를 아예 못 했다.** 순간 상태가 아니라 그 상태로
             **얼마나 갔는지**를 세야 했다.
       집행: 팔이 **자기 위치를** 스프링-댐퍼로 붙든다. 채널을 거치지 않으므로 채널이 죽어도
             동작하고, 고정점을 향한 소산이라 에너지를 만들지 않는다.
             (exp 50 의 "가상 벽은 로컬에서 렌더링해야 한다"를 **실패 경로**에 적용한 것이다.)
     결과: 맹행 흔들림이 **1.17 ~ 2.75 mm** 로 좁아지고 과제는 계속 완주한다. 경계가 어느 항이
     살아 있는지와 무관해졌다 — **설계에서 온다.**
  C. **그러면 exp 57 에서 실패한 처방이 이제 되는지 확인한다.** 브레이크를 대체했으니 λ 를 예산으로
     죄어도 손해가 없어야 한다. 결과: exp 57 에서 4.72 → 6.53 mm 로 악화됐던 것이, 정지를 먼저
     넣으면 1.91 → **1.68 mm** 로 해가 되지 않는다. exp 57 이 규칙으로만 적어둔 순서
     (**대체 먼저, 억제 나중** = VERIFICATION R20)를 숫자로 확인한 것이다.

대가도 같이 잰다. 두 개다.
  · **선언한 여유가 정지 빈도를 산다**: 여유 0.5 mm 면 시간의 56% 를 멈춰 있고, 4.0 mm 면 12% 다.
    교환비가 통신이 아니라 **해부**에서 온다 — 같은 코드·같은 네트워크로 시술마다 달라진다.
  · **복귀가 가장 위험한 순간이다**: 즉시 복귀는 155 mm/s 로 튀어나가고, 램프를 200 ms 로 늘리면
    ~90 mm/s 인데 정지 구간이 9% → 73% 가 된다(exp 47 의 관통 돌진과 같은 모양).

    python scripts/58_stop_when_lost.py
"""

from __future__ import annotations

import sys
from functools import partial
from importlib import import_module
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
jc = import_module("56_jittery_channel")
bc = import_module("57_bursty_channel")

DT = jc.DT
X_TARGET = jc.X_TARGET
X_SURFACE = jc.X_SURFACE
LAM_TASK = jc.LAM_TASK
TAIL_MS, BURST_LEN = bc.TAIL_MS, bc.BURST_LEN
N_SEEDS = 6

# 이 실험의 기준 채널 — exp 57 의 최악 조건(긴 꼬리 + 긴 연집)에서 시험한다.
CHAN = dict(tail_ms=TAIL_MS, loss=0.10, burst_len=160.0)
RESUME_MS = 60.0          # 정지 해제 램프. E 절에서 스윕해 돌진과의 교환비를 본다
BLIND_MM = 1.0            # **선언된 임상 여유** [mm] — exp 45 shaft 2.17 / exp 48 통로 1.25 계열.


def run(mode="tdpa", seed=0, **kw):
    """exp 57 의 채널 + exp 56 의 플랜트. kw 는 그대로 전달된다."""
    ch = dict(CHAN)
    ch.update({k: kw.pop(k) for k in list(kw) if k in ("tail_ms", "loss", "burst_len")})
    kw.setdefault("blind_mm", BLIND_MM)
    return bc.run(mode=mode, seed=seed, **ch, **kw)


def sweep(mode="tdpa", seeds=N_SEEDS, **kw):
    rs = [run(mode=mode, seed=s, **kw) for s in range(seeds)]
    ok = [r for r in rs if not r["diverged"]]

    def med(key, default=np.nan):
        if not ok:
            return default
        v = np.asarray([r[key] for r in ok], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else np.nan

    return dict(runs=rs, n=len(rs), n_div=sum(r["diverged"] for r in rs),
                blind_max_mm=med("blind_max_mm"), blind_mm=med("blind_mm"),
                final_depth_mm=med("final_depth_mm"), osc_mm=med("osc_mm", np.inf),
                pos_err_mm=med("pos_err_mm", np.inf),
                e_min=(float(np.min([r["e_min"] for r in ok])) if ok else -np.inf),
                held_frac=float(np.mean([r["held_frac"] for r in rs])),
                n_estop=float(np.mean([r["n_estop"] for r in rs])),
                resume_vmax_mms=med("resume_vmax_mms"),
                max_starve_ms=med("max_starve_ms"),
                hold_early_um=med("hold_early_um"), hold_late_um=med("hold_late_um"))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    print("=== 58. 끊기면 멈추기: exp 57 이 남긴 '미해결'을 설계해 닫는다 ===")
    print(f"채널은 exp 57 의 최악 조건 그대로 — 파레토 꼬리(x_m={TAIL_MS:.0f} ms) + 손실 10% + "
          f"평균 연집 160 ms. 플랜트·제어는 exp 56.")
    print(f"정지 판정 기준은 **선언된 여유 {BLIND_MM:.1f} mm**(exp 45 shaft 2.17 / exp 48 통로 "
          f"1.25 계열) — 통신 파라미터가 아니다.")

    cases = [
        ("전부 있음 (exp 57 조건)", dict()),
        ("− 조직 반력 (자유공간 접근)", dict(tissue_on=False)),
        ("− 표류 보정 λ (exp 50 의 파동)", dict(lam_pos=0.0)),
        ("− 국소 감쇠 ×0.5", dict(b_scale=0.5)),
        ("− 국소 감쇠 ×0.3", dict(b_scale=0.3)),
        ("− 조직 − λ (둘 다)", dict(tissue_on=False, lam_pos=0.0)),
    ]
    if quick:
        cases = [cases[0], cases[2], cases[4]]

    # ---------------- A ----------------
    print("-" * 96)
    print("[A] 절제 — 끊긴 동안 도구를 세우는 것이 무엇인가 (정지 없음, ZOH)")
    print(f"{'구성':>30s} | {'한 구간 맹행[mm]':>15s} | {'홀드 초반→후반[㎛]':>18s} | "
          f"{'최대 굶음[ms]':>12s} | {'도달깊이[mm]':>11s}")
    A = {}
    for lbl, kw in cases:
        s = sweep("zoh", seeds=seeds, **kw)
        A[lbl] = s
        e, l = s["hold_early_um"], s["hold_late_um"]
        arrow = f"{e:.0f}→{l:.0f}" if np.isfinite(l) else f"{e:.0f}→—"
        print(f"{lbl:>30s} | {s['blind_max_mm']:15.2f} | {arrow:>18s} | "
              f"{s['max_starve_ms']:12.0f} | {s['final_depth_mm']:11.1f}")
    off = [A[l]["blind_max_mm"] for l, _ in cases]
    print(f"  같은 채널·같은 암전인데 맹행이 **{min(off):.2f} ~ {max(off):.2f} mm** 로 벌어진다.")
    print("  홀드 초반→후반 이동량이 줄어드는 것이 exp 57 이 찾은 '자기 제한'인데, 국소 감쇠를 깎으면")
    print("  그 감쇠가 옅어진다. **즉 지금의 경계는 어느 항이 남아 있느냐에 달려 있다** — 설계가 아니라")
    print("  이 플랜트가 우연히 갖고 있던 성질이다. exp 57 이 '일반화 금지'라고 적은 근거가 이 표다.")

    # ---------------- B ----------------
    print("-" * 96)
    print("[B] 정지를 설계해 넣는다 — 판정은 선언된 여유, 집행은 국소 위치 유지")
    print(f"{'구성':>30s} | {'정지없음[mm]':>11s} | {'정지[mm]':>9s} | {'감축':>6s} | "
          f"{'정지 구간':>8s} | {'정지 횟수':>8s} | {'도달깊이[mm]':>11s}")
    B = {}
    for lbl, kw in cases:
        on = sweep("tdpa", seeds=seeds, estop=True, resume_ms=RESUME_MS, **kw)
        B[lbl] = on
        r = A[lbl]["blind_max_mm"] / max(on["blind_max_mm"], 1e-9)
        print(f"{lbl:>30s} | {A[lbl]['blind_max_mm']:11.2f} | {on['blind_max_mm']:9.2f} | "
              f"{r:5.1f}× | {on['held_frac']*100:7.1f}% | {on['n_estop']:8.1f} | "
              f"{on['final_depth_mm']:11.1f}")
    on_v = [B[l]["blind_max_mm"] for l, _ in cases]
    print(f"  **핵심은 평균이 아니라 흔들림이 좁아진 것이다**: {min(off):.2f}~{max(off):.2f} → "
          f"{min(on_v):.2f}~{max(on_v):.2f} mm.")
    print(f"  경계가 이제 **선언한 여유({BLIND_MM:.1f} mm)에서 온다** — 어느 항이 살아 있는지와 무관하다.")
    print(f"  선언값의 약 2배로 묶이는데, 그 초과분이 **정지 거리**다(트리거가 걸린 뒤 붙잡는 동안 더")
    print("  가는 몫). 그것도 플랜트 성질이라 여유를 정할 때 함께 봐야 한다.")
    print("  그리고 **과제는 계속 완주한다** — 정지가 '안전하지만 못 쓰는' 쪽으로 퇴화하지 않았다.")

    # ---------------- C ----------------
    print("-" * 96)
    print("[C] exp 57 의 실패한 처방 재시험 — 브레이크를 대체했으니 λ 를 죄어도 되는가")
    print(f"{'구성':>30s} | {'맹행[mm]':>9s} | {'E_min[mJ]':>10s} | {'도달깊이[mm]':>11s} | "
          f"{'진동[mm]':>8s}")
    C = {}
    arms = [("예산만 (exp 56)", dict()),
            ("예산 + λ 게이트 (exp 57: 실패)", dict(lam_gate=True)),
            ("예산 + 정지", dict(estop=True, resume_ms=RESUME_MS)),
            ("예산 + 정지 + λ 게이트", dict(estop=True, resume_ms=RESUME_MS,
                                            lam_gate=True))]
    for lbl, kw in arms:
        s = sweep("tdpa", seeds=seeds, **kw)
        C[lbl] = s
        print(f"{lbl:>30s} | {s['blind_max_mm']:9.2f} | {s['e_min']*1e3:10.4f} | "
              f"{s['final_depth_mm']:11.1f} | {s['osc_mm']:8.3f}")
    a_, b_, c_, d_ = (C[l]["blind_max_mm"] for l, _ in arms)
    print(f"  exp 57 이 재현된다: 예산만 {a_:.2f} → λ 게이트를 얹으면 {b_:.2f} mm (**악화**).")
    print(f"  정지를 먼저 넣으면 {c_:.2f}, 그 위에 같은 λ 게이트를 얹으면 {d_:.2f} mm.")
    if d_ <= max(c_ * 1.15, c_ + 0.2):
        print("  → **더 이상 해가 되지 않는다.** 브레이크를 대체한 뒤에는 그 항을 죄어도 잃을 것이 없다.")
        print("  exp 57 이 규칙으로만 적어둔 순서(**대체 먼저, 억제 나중**)를 숫자로 확인한 것이다.")
    else:
        print("  → **여전히 손해다.** 정지가 그 항의 역할을 온전히 대체하지 못했다(정직한 결과) —")
        print("  대체물이 있어도 억제의 대가가 남는다면, 그 항은 하나 이상의 일을 하고 있는 것이다.")

    # ---------------- D ----------------
    print("-" * 96)
    print("[D] 선언한 여유가 **정지 빈도를 산다** — 이게 이 설계의 실제 손잡이다")
    print(f"{'여유[mm]':>9s} | {'맹행[mm]':>9s} | {'선언값 대비':>10s} | {'정지 구간':>8s} | "
          f"{'정지 횟수':>8s} | {'도달깊이[mm]':>11s}")
    D = {}
    for bm in ([0.5, 2.0] if quick else [0.5, 1.0, 2.0, 4.0]):
        s = sweep("tdpa", seeds=seeds, estop=True, resume_ms=RESUME_MS, blind_mm=bm)
        D[bm] = s
        print(f"{bm:9.1f} | {s['blind_max_mm']:9.2f} | "
              f"{s['blind_max_mm']/bm:9.1f}× | {s['held_frac']*100:7.1f}% | "
              f"{s['n_estop']:8.1f} | {s['final_depth_mm']:11.1f}")
    lo, hi = min(D), max(D)
    print(f"  여유를 {lo:.1f} → {hi:.1f} mm 로 늘리면 정지 구간이 "
          f"{D[lo]['held_frac']*100:.0f}% → {D[hi]['held_frac']*100:.0f}% 로 줄고 맹행은 "
          f"{D[lo]['blind_max_mm']:.2f} → {D[hi]['blind_max_mm']:.2f} mm 로 늘어난다.")
    print("  **교환비가 통신이 아니라 해부에서 온다는 게 요점이다.** 통로가 좁은 시술은 자주 멈추고,")
    print("  여유가 큰 시술은 거의 멈추지 않는다 — 같은 코드, 같은 네트워크로.")

    # ---------------- E ----------------
    print("-" * 96)
    print("[E] 공짜가 아니다 — 정지를 풀 때의 돌진 (복귀 램프 스윕, 해제 직후 100 ms 관찰)")
    print(f"{'복귀 램프[ms]':>13s} | {'복귀 최대속도[mm/s]':>18s} | {'맹행[mm]':>9s} | "
          f"{'정지 구간':>8s} | {'도달깊이[mm]':>11s}")
    E = {}
    for rm in ([0.0, 200.0] if quick else [0.0, 20.0, 60.0, 120.0, 200.0]):
        s = sweep("tdpa", seeds=seeds, estop=True, resume_ms=rm)
        E[rm] = s
        print(f"{rm:13.0f} | {s['resume_vmax_mms']:18.1f} | {s['blind_max_mm']:9.2f} | "
              f"{s['held_frac']*100:7.1f}% | {s['final_depth_mm']:11.1f}")
    r0, r1 = min(E), max(E)
    print(f"  즉시 복귀는 {E[r0]['resume_vmax_mms']:.0f} mm/s 로 튀어나가고, 램프를 {r1:.0f} ms 로 "
          f"늘리면 {E[r1]['resume_vmax_mms']:.0f} mm/s 다. 대신 정지 구간이 "
          f"{E[r0]['held_frac']*100:.0f}% → {E[r1]['held_frac']*100:.0f}% 로 늘어난다.")
    print("  exp 47 의 관통 돌진과 같은 모양이고 같은 결론이다 — **작동점은 임상 제약이 고른다.**")
    print("  램프에는 부수적으로 좋은 성질이 하나 있다: 램프는 **정보가 오는 스텝에서만** 올라가므로")
    print("  링크가 나쁠수록 복귀가 느려진다. 링크 품질에 비례한 조심성이 공짜로 붙는다.")

    # ---------------- F ----------------
    print("-" * 96)
    print("[F] 남은 것 — 정직하게")
    print("  · 정지는 팔을 **그 자리에** 세운다. 조직 안이라면 도구가 박힌 채 멈추는 것이고, 안전한")
    print("    상태로 **후퇴**하는 것이 아니다. 후퇴 정책은 임상 판단이고 여기서 다루지 않았다.")
    print("  · 판정에 통신 임계값은 없지만 **선언된 여유**와 **복귀 램프**는 여전히 선택이다. 다만 둘 다")
    print("    네트워크가 아니라 해부·액추에이터에서 오는 숫자라, 고르는 사람이 답을 갖고 있다.")
    print("  · 술자 쪽은 그대로다 — 팔만 멈추고 마스터는 계속 움직인다. 그래서 복귀 시 어긋남이 크고,")
    print("    실제 시스템은 마스터도 잠그거나 힘으로 알린다. 그건 사람이 든 루프라 별도 실험이 필요하다.")
    print("  · 이 정지는 **에너지 예산이 옳다는 전제** 위에 서 있다(β 로 마름을 판정한다). 장부가 틀리면")
    print("    정지도 틀린다 — exp 53 의 '검산도 같은 센서로 하면' 과 같은 종류의 의존이다.")

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    en_all = ["all present", "− tissue", "− drift term", "− damping ×0.5",
              "− damping ×0.3", "− tissue − drift"]
    en = en_all if not quick else [en_all[0], en_all[2], en_all[4]]

    ax = axes[0, 0]
    x = np.arange(len(cases))
    ax.bar(x, off, color="crimson")
    ax.set_xticks(x); ax.set_xticklabels(en, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("worst blind tool travel [mm]")
    ax.set_title("Without a stop, the bound depends on which term survives",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    ax.bar(x - 0.2, off, 0.4, color="crimson", label="no stop")
    ax.bar(x + 0.2, on_v, 0.4, color="tab:blue", label="with the local stop")
    ax.axhline(BLIND_MM, color="0.3", ls="--", lw=1, label="declared margin")
    ax.set_xticks(x); ax.set_xticklabels(en, fontsize=7, rotation=25, ha="right")
    ax.set_ylabel("worst blind tool travel [mm]")
    ax.set_title("A designed bound does not care which term survives", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=7)

    ax = axes[0, 2]
    names = ["budget\nonly", "+ gate λ\n(exp 57)", "+ stop", "+ stop\n+ gate λ"]
    xa = np.arange(4)
    ax.bar(xa, [C[l]["blind_max_mm"] for l, _ in arms],
           color=["0.6", "crimson", "tab:blue", "seagreen"])
    ax.set_xticks(xa); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("worst blind tool travel [mm]")
    ax.set_title("Replace first, then you may suppress — exp 57's rule, tested",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    bms = sorted(D)
    ax.plot(bms, [D[b]["held_frac"] * 100 for b in bms], "-o", color="tab:blue",
            label="time stopped [%]")
    ax2 = ax.twinx()
    ax2.plot(bms, [D[b]["blind_max_mm"] for b in bms], "-s", color="crimson")
    ax2.plot(bms, bms, ":", color="0.5", lw=1)
    ax2.set_ylabel("blind travel [mm] (dotted = declared)", color="crimson",
                   fontsize=8)
    ax.set_xlabel("declared clinical margin [mm]")
    ax.set_ylabel("time stopped [%]", color="tab:blue")
    ax.set_title("The margin buys the stop rate — set by anatomy, not the network",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1, 1]
    for lbl, kw, c in [("hold last (no stop)", dict(mode="zoh"), "crimson"),
                       ("energy budget", dict(mode="tdpa"), "0.55"),
                       ("budget + local stop",
                        dict(mode="tdpa", estop=True, resume_ms=RESUME_MS),
                        "tab:blue")]:
        r = run(seed=0, **kw)
        ax.plot(r["log"]["t"], r["log"]["xs"] * 1e3, color=c, lw=1.3, label=lbl)
    r_stop = run("tdpa", seed=0, estop=True, resume_ms=RESUME_MS)
    held = r_stop["log"]["held"] > 0.5
    ax.fill_between(r_stop["log"]["t"], 0, 60, where=held, color="tab:blue",
                    alpha=0.12, label="stopped")
    ax.axhline(X_TARGET * 1e3, color="0.3", ls="--", lw=1)
    ax.axhline(X_SURFACE * 1e3, color="0.6", ls=":", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tool depth [mm]"); ax.set_ylim(0, 60)
    ax.set_title("What the stop looks like on the task", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1, 2]
    rms = sorted(E)
    ax.plot(rms, [E[r]["resume_vmax_mms"] for r in rms], "-o", color="crimson",
            label="peak speed after release [mm/s]")
    ax2 = ax.twinx()
    ax2.plot(rms, [E[r]["held_frac"] * 100 for r in rms], "-^", color="tab:blue")
    ax2.set_ylabel("time stopped [%]", color="tab:blue", fontsize=9)
    ax.set_xlabel("resume ramp [ms]")
    ax.set_ylabel("peak speed [mm/s]", color="crimson")
    ax.set_title("Lunge versus lost time (exp 47's shape, again)", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    fig.suptitle("58. Stop when the link is lost — replacing the brake nobody designed",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "58_stop_when_lost.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/58_stop_when_lost.png, assets/58_stop_when_lost.png")

    return dict(A=A, B=B, C=C, D=D, E=E)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - 정지는 팔을 그 자리에 붙든다. **후퇴(retract)가 아니다.** 조직 안에서의 안전 상태가 '정지'인지
#     '후퇴'인지는 임상 판단이고, 여기서는 정하지 않았다.
#   - 통신 임계값은 없앴지만 **선언된 여유**와 **복귀 램프**는 여전히 선택이다. 다만 둘 다 해부·
#     액추에이터에서 오는 숫자라 고르는 사람이 근거를 갖고 있다는 점이 다르다.
#   - 마스터는 계속 움직인다(술자를 모델화된 임피던스로 두었으므로). 실제 시스템은 정지 중 마스터도
#     잠그거나 힘으로 알린다 — 사람이 든 루프라 별도 실험이 필요하다.
#   - 판정이 β(에너지 예산)에 의존한다. 장부가 틀리면 정지 시점도 틀린다.
#   - 국소 유지 이득 K_HOLD·D_HOLD 를 팔의 국소 감쇠와 같은 자릿수로 고정했다. 튜닝하지 않았지만,
#     관성이 아주 작은 축에서는 이 이득이 그대로는 과할 수 있다(exp 47 의 바늘 스핀 계열).
#   - 절제에서 국소 감쇠를 ×0.1 까지 내리면 정지 여부와 무관하게 **시스템 자체가 발산**한다.
#     그건 암전 문제가 아니라 제어 설계 문제라 절제 범위를 ×0.3 까지로 잡았다.
#   - exp 56·57 과 같은 채널·플랜트 이상화(2상태 연집, 독립 양방향, 계획 궤적을 따르는 술자)를
#     그대로 물려받는다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
