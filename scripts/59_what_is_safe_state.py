"""원격조작 V: 정지가 안전 상태인가 — 물어보려면 모델을 먼저 고쳐야 했다.

exp 58 이 통신 상실 정지를 설계하고 자백 두 개를 남겼다.
  · 정지는 팔을 **그 자리에** 붙든다. 조직 안이면 **박힌 채** 멈추는 것이고 안전한 상태로 **후퇴**
    하는 것이 아니다. 어느 쪽이 옳은지는 정하지 않았다.
  · **술자 쪽은 그대로다.** 팔만 멈추고 마스터는 계속 움직여서 복귀가 155 mm/s 로 튄다.

둘을 물으려다 **두 번 막혔고, 막힌 이유가 이 실험의 결과다. 둘 다 제어 문제가 아니라 모델 문제였다.**

--- 결론 요약 ---
1. **물어보려면 조직 모델을 먼저 고쳐야 했다.** "붙들고 있으면 위험한가"를 물으려면 정지 중에 환자가
   움직여야 한다. 넣어 봤더니 힘이 거의 안 변한다 — 사슬이 exp 47 이후 쓴 조직 모델은 관통 후
   **절삭력 + 마찰**뿐이라 **탄성이 없어서** 조직이 다가오는 것을 힘으로 표현하지 못한다.
   도구를 고정하고 표면만 흔드는 최소 시험에서 힘 변동이 5 mm 왕복에 **0.12 N**이다.
   바늘-조직 마찰의 표준 모형인 **stick-slip 파악**을 넣으면 같은 조건에서 **1.62 N — 14배**.
   → **모델이 표현하지 못하는 위해는 정책으로 비교할 수 없다. '차이 없음'이 아니라 침묵이었다.**
2. **그런데 물어보니 후퇴가 이기지 않는다**(정직한 네거티브). 파악 항의 몫은 미끄러짐 한계에서
   포화하고(≈2×F_slip) 절삭 기저(2~3 N)에 비해 작다. 후퇴는 조직에 얹는 몫을 1.68 → 0.89 N 로
   줄이지만 맹행을 **3.6배**(2.01 → 7.28 mm) 늘린다 — 후퇴 자체가 정보 없이 하는 움직임이다.
   뒤집힐 조건은 명확하다: 미끄러짐 한계가 크거나 절삭 기저가 작으면 — **어느 조직인가가 정책을
   고른다.** 그 값은 실측에서 와야 한다.
3. **술자 쪽에서는 대책의 부호가 뒤집힌다.** 마스터를 잠그면 복귀가 **나빠진다**(120 → 133 mm/s).
   잠금이 술자의 의도를 없애는 게 아니라 **손 스프링에 저장**하고, 더 나쁘게는 **술자가 반응해야 할
   그 신호(손과 도구의 어긋남)를 가린다.** 진짜 이득은 술자가 반응할 수 있게 하는 쪽이다 —
   "도구가 뒤처지면 반응시간 뒤에 손을 멈춘다" 규칙 하나로 복귀 중앙값이 **120 → 68 mm/s** —
   단 **같은 시드끼리 짝지으면 감소율 중앙값 25%, 개선된 시드 11/16** 이다(보편적이지 않다).
   중앙값 대 중앙값은 두 분포가 치우쳐 있어 짝지은 효과를 과장한다 — exp 52 의 검출률↔AUROC 와
   같은 종류의 실수라 짝지은 통계를 같이 낸다.
   → **술자 쪽 대책은 술자 모델이 적응적일 때만 평가할 수 있다.** exp 50 이 한계로 적어둔
   "술자는 고정 임피던스"가 여기서 **결론의 부호를 바꾸는** 크기로 작용했다.
4. 권장 조합은 **붙들기 + 술자 반응**이다(맹행 1.80 mm, 복귀 중앙값 68 mm/s, 수동성·완주 유지).

    python scripts/59_what_is_safe_state.py
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
s58 = import_module("58_stop_when_lost")

DT = jc.DT
X_SURFACE, X_TARGET = jc.X_SURFACE, jc.X_TARGET
TAIL_MS = bc.TAIL_MS
N_SEEDS = 6

BURST_MS = 160.0          # exp 57·58 의 최악 조건
BREATH_MM = 5.0           # 환자 움직임 진폭 — 복부·흉부 호흡 계열(뇌는 프레임 고정으로 훨씬 작다)
BREATH_HZ = 0.25
RETRACT_MM = 5.0
REACT_MS = 200.0          # 술자 반응시간

# stick-slip 파악. 조직이 축을 붙잡고 있다가 탄성력이 한계를 넘으면 미끄러진다.
K_GRIP = 300.0            # [N/m] 파악 강성
F_SLIP = 0.8              # [N] 미끄러짐 한계 — 이 값을 넘으면 조직이 축을 놓아준다


class GrippingTissue(jc.tele.Tissue):
    """exp 47/50 의 절삭+마찰 모델에 **stick-slip 파악** 항을 더한 것.

    바늘-조직 상호작용의 표준 모형이다. 축이 움직이는 동안은 계속 미끄러져 기존 모델로 환원되고,
    **멈춰 있는 동안은 조직이 축에 붙어** 상대 변위를 탄성으로 받는다 — 그래서 "도구를 붙들고 있는데
    환자가 움직인다"가 비로소 힘으로 나타난다. 기존 모델에는 이 항이 없어서 그 질문을 물을 수 없었다.
    """

    def __init__(self, k_grip=K_GRIP, f_slip=F_SLIP):
        super().__init__()
        self.k, self.f_slip = k_grip, f_slip
        self.anchor = None

    def force(self, x):
        base = super().force(x)          # 절삭 전 비선형 강성 → 관통 → 절삭+마찰 (부호: 음수=저항)
        if not self.punctured:
            self.anchor = x
            return base
        if self.anchor is None:
            self.anchor = x
        f_el = self.k * (x - self.anchor)
        if abs(f_el) > self.f_slip:      # 한계를 넘으면 조직이 놓아준다(앵커가 끌려온다)
            self.anchor = x - np.sign(f_el) * self.f_slip / self.k
            f_el = np.sign(f_el) * self.f_slip
        return base - f_el


def hold_and_breathe(grip, breath_mm=BREATH_MM, depth_mm=12.0, secs=4.0):
    """**조직 모델만** 단독으로 시험한다 — 채널도 제어도 없다.

    도구를 관통 후 깊이까지 밀어 넣고 **그 자리에 고정**한 뒤 표면을 주기적으로 움직인다.
    "붙들고 있는데 환자가 움직인다"를 모델이 힘으로 표현하는지 보는 최소 시험이다.
    전체 시뮬로는 안 보인다 — 정지 직후에도 도구가 스스로 더 파고드는 몫이 이 효과를 덮는다.
    """
    ts = jc.tele.Tissue() if not grip else GrippingTissue()
    x = X_SURFACE
    # 관통시킬 만큼 천천히 밀어 넣는다(관통 플래그가 서야 파악 항이 의미를 갖는다)
    for _ in range(4000):
        x += depth_mm * 1e-3 / 4000.0
        ts.force(x)
    f = []
    for k in range(int(secs / DT)):
        surf = breath_mm * 1e-3 * np.sin(2 * np.pi * BREATH_HZ * k * DT)
        f.append(abs(ts.force(x - surf)))
    f = np.asarray(f)
    return float(f.max() - f.min()), float(f.mean())


def run(seed=0, grip=False, react=False, **kw):
    """exp 57 의 채널 + exp 56 의 플랜트. grip/react 로 모델 간극을 켠다."""
    p = dict(tail_ms=TAIL_MS, loss=0.10, burst_len=BURST_MS,
             estop=True, resume_ms=60.0, blind_mm=1.0,
             breath_mm=BREATH_MM, breath_hz=BREATH_HZ)
    p.update(kw)
    if grip:
        p["tissue_obj"] = GrippingTissue()
    if react:
        p.setdefault("op_react_ms", REACT_MS)
    return bc.run("tdpa", seed=seed, **p)


def sweep(seeds=N_SEEDS, **kw):
    rs = [run(seed=s, **kw) for s in range(seeds)]
    ok = [r for r in rs if not r["diverged"]]

    def med(key):
        if not ok:
            return np.nan
        v = np.asarray([r[key] for r in ok], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else np.nan

    return dict(runs=rs, n=len(rs), n_div=sum(r["diverged"] for r in rs),
                f_held=med("f_e_held_max"), df_held=med("df_held_max"),
                depth_held=med("depth_held_max_mm"),
                blind=med("blind_max_mm"), final=med("final_depth_mm"),
                mism=med("mismatch_release_mm"), vres=med("resume_vmax_mms"),
                osc=med("osc_mm"), held=float(np.mean([r["held_frac"] for r in rs])),
                e_min=(float(np.min([r["e_min"] for r in ok])) if ok else -np.inf))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(quick=False):
    seeds = 2 if quick else N_SEEDS
    print("=== 59. 정지가 안전 상태인가 — 물어보려면 모델을 먼저 고쳐야 했다 ===")
    print(f"채널·정지는 exp 57·58 그대로(파레토 꼬리, 손실 10%, 평균 연집 {BURST_MS:.0f} ms, "
          f"여유 1.0 mm). 새로 넣은 것은 **환자 움직임**({BREATH_MM:.0f} mm / {BREATH_HZ:.2f} Hz)과")
    print("조직의 **stick-slip 파악**, 그리고 **적응형 술자**다.")

    # ---------------- A. 모델이 그 위해를 표현하는가 (조직 모델 단독) ----------------
    print("-" * 96)
    print("[A] 먼저 물어야 할 것 — 사슬이 쓴 조직 모델이 '붙들고 있는 위해'를 표현하는가")
    print("    도구를 관통 후 깊이에 **고정**하고 표면만 흔든다. 채널도 제어도 없는 최소 시험이다")
    print("    (전체 시뮬로는 안 보인다 — 정지 직후에도 도구가 스스로 더 파고드는 몫이 덮는다).")
    print(f"{'환자 움직임[mm]':>14s} | {'절삭+마찰: 힘 변동[N]':>20s} | "
          f"{'+ stick-slip: 힘 변동[N]':>23s} | {'비':>6s} | {'파악 항의 몫[N]':>14s}")
    A = {}
    brs = [0.0, 5.0, 20.0] if quick else [0.0, 2.0, 5.0, 10.0, 20.0]
    for br in brs:
        s0, _ = hold_and_breathe(False, breath_mm=br)
        s1, _ = hold_and_breathe(True, breath_mm=br)
        A[br] = (s0, s1)
        ratio = s1 / s0 if s0 > 1e-9 else np.inf
        print(f"{br:14.0f} | {s0:20.3f} | {s1:23.3f} | "
              f"{(f'{ratio:.1f}×' if np.isfinite(ratio) else '—'):>6s} | {s1 - s0:14.3f}")
    b5 = A.get(5.0, A[brs[-1]])
    print(f"  절삭+마찰 모델에서는 표면이 5 mm 왕복해도 힘이 {b5[0]:.2f} N 밖에 안 움직인다"
          "(마찰항 MU·d 뿐).")
    print("  **관통 후 탄성이 없어서 조직이 다가오는 것을 힘으로 표현하지 못한다.**")
    print(f"  파악 항을 넣으면 같은 조건에서 {b5[1]:.2f} N — {b5[1]/max(b5[0],1e-9):.0f}배다.")
    print(f"  오른쪽 칸(두 모델의 차 = 파악 항의 순수 몫)이 5~10 mm 에서 "
          f"**{2*F_SLIP:.1f} N(=2×미끄러짐 한계) 근처에서 포화**한다 — 조직이 놓아주기 때문이다.")
    print("  (20 mm 에서 차가 다시 작아지는 것은 그쯤이면 기저 마찰 변동 자체가 커져서다.)")
    print("  → **모델이 표현하지 못하는 위해는 정책으로 비교할 수 없다.** exp 58 의 '붙들기 vs 후퇴'는")
    print("  제어 문제로 보였지만 먼저 **모델 표현력 문제**였다.")

    # ---------------- B. 그래도 정책 수준에서는 후퇴가 이기지 않는다 ----------------
    print("-" * 96)
    print("[B] 이제 물을 수 있다 — 그런데 후퇴가 이기지 않는다 (파악 항 켠 상태)")
    print(f"{'정지 정책':>18s} | {'정지중 최대힘[N]':>15s} | {'붙들며 더 얹은 몫[N]':>18s} | "
          f"{'한 구간 맹행[mm]':>15s} | {'도달깊이[mm]':>11s}")
    B = {}
    for lbl, rm in (("붙들기(exp 58)", 0.0), (f"후퇴 {RETRACT_MM:.0f} mm", RETRACT_MM)):
        s = sweep(seeds=seeds, grip=True, retract_mm=rm)
        B[lbl] = s
        print(f"{lbl:>18s} | {s['f_held']:15.2f} | {s['df_held']:18.3f} | "
              f"{s['blind']:15.2f} | {s['final']:11.1f}")
    hb, rb = B["붙들기(exp 58)"], B[f"후퇴 {RETRACT_MM:.0f} mm"]
    print(f"  후퇴는 맹행을 {hb['blind']:.2f} → {rb['blind']:.2f} mm 로 "
          f"{rb['blind']/max(hb['blind'],1e-9):.1f}배 늘리면서 조직에 얹는 몫을 "
          f"{hb['df_held']:.2f} → {rb['df_held']:.2f} N 로 줄인다.")
    print("  **정직한 네거티브다**: A 절에서 본 대로 파악 항의 몫은 미끄러짐 한계에서 유계이고, 그 값이")
    print("  절삭 기저(2~3 N)에 비해 작다. 그래서 후퇴로 사는 것이 **후퇴 자체가 정보 없이 하는")
    print("  움직임**이라는 대가를 넘지 못한다. 이 모델·이 파라미터에서는 **붙들기가 맞다.**")
    print("  뒤집힐 조건은 명확하다: 미끄러짐 한계가 크거나(조직이 잘 안 놓아준다) 절삭 기저가 작으면")
    print("  — 즉 **어느 조직인가**가 정책을 고른다. 그 값은 실측에서 와야 한다.")

    # ---------------- C. 술자 쪽 ----------------
    print("-" * 96)
    print("[C] exp 58 의 두 번째 자백 — 술자 쪽. 여기가 실제로 이득이 있는 곳이다")
    print(f"{'술자 모델':>18s} | {'마스터':>12s} | {'해제 시 어긋남[mm]':>17s} | "
          f"{'복귀 최대속도[mm/s]':>18s} | {'정지 구간':>8s} | {'도달깊이[mm]':>11s}")
    C = {}
    for ol, react in (("고정 임피던스(exp 50~58)", False), (f"+ 반응 {REACT_MS:.0f} ms", True)):
        for ml_lbl, ml in (("자유(exp 58)", False), ("국소 제동", True)):
            s = sweep(seeds=seeds, grip=True, react=react, master_lock=ml)
            C[(ol, ml_lbl)] = s
            print(f"{ol if not ml else '':>18s} | {ml_lbl:>12s} | {s['mism']:17.2f} | "
                  f"{s['vres']:18.1f} | {s['held']*100:7.1f}% | {s['final']:11.1f}")
    b0 = C[("고정 임피던스(exp 50~58)", "자유(exp 58)")]
    l0 = C[("고정 임피던스(exp 50~58)", "국소 제동")]
    rk = f"+ 반응 {REACT_MS:.0f} ms"
    r0 = C[(rk, "자유(exp 58)")]
    r1 = C[(rk, "국소 제동")]
    print(f"  **마스터 제동은 두 술자 모델 모두에서 복귀를 나쁘게 한다**"
          f"({b0['vres']:.0f} → {l0['vres']:.0f}, {r0['vres']:.0f} → {r1['vres']:.0f} mm/s).")
    print("  어긋남은 줄어드는데(잠갔으니 당연히) 복귀는 더 빠르다 — 잠금이 술자의 의도를 없애는 게")
    print("  아니라 **손 스프링에 저장**하고, 게다가 **술자가 반응해야 할 그 신호(손과 도구의 어긋남)를")
    print("  가려버린다.** 대책이 그 대책이 필요한 이유를 숨기는 구조다.")
    print(f"  진짜 이득은 술자가 반응할 수 있게 하는 쪽에 있다: 반응 규칙만으로 복귀 중앙값이 "
          f"{b0['vres']:.0f} → {r0['vres']:.0f} mm/s.")
    # **중앙값 비교는 짝지은 효과를 과장한다.** 두 분포가 다 오른쪽으로 치우쳐 있어서
    # median(A)/median(B) 가 시드별 개선율보다 크게 나온다 — exp 52 에서 고정 오경보 검출률을
    # AUROC 로 바꿔야 했던 것과 같은 종류의 실수다. 그래서 같은 시드끼리 짝지어 다시 센다.
    fx = np.array([r["resume_vmax_mms"] for r in b0["runs"] if not r["diverged"]])
    rc = np.array([r["resume_vmax_mms"] for r in r0["runs"] if not r["diverged"]])
    n = min(len(fx), len(rc))
    if n:
        red = (fx[:n] - rc[:n]) / np.maximum(fx[:n], 1e-9) * 100
        print(f"  단, **같은 시드끼리 짝지어 보면 감소율 중앙값 {np.median(red):.0f}%, "
              f"개선된 시드 {int((red > 0).sum())}/{n}** 이다 — 보편적이지 않다.")
        print("  중앙값 대 중앙값(즉 1.8배)은 두 분포가 다 오른쪽으로 치우쳐 있어서 **짝지은 효과를")
        print("  과장한다.** exp 52 에서 고정 오경보 검출률을 AUROC 로 바꿔야 했던 것과 같은 실수다.")
    print("  → **술자 쪽 대책은 술자 모델이 적응적일 때만 평가할 수 있다.** exp 50 이 '술자는 고정")
    print("  임피던스 모델'이라고 적어둔 한계가 여기서 **결론의 부호를 바꾸는** 크기로 작용했다.")

    # ---------------- D. 합쳐서 ----------------
    print("-" * 96)
    print("[D] 합친 정책 — exp 58 의 성질(경계·수동성·완주)이 유지되는지")
    print(f"{'구성':>30s} | {'맹행[mm]':>9s} | {'얹은 몫[N]':>11s} | {'복귀[mm/s]':>11s} | "
          f"{'E_min[mJ]':>10s} | {'도달깊이[mm]':>11s}")
    D = {}
    combos = [("exp 58 그대로", dict()),
              ("+ 술자 반응", dict(react=True)),
              ("+ 술자 반응 + 후퇴", dict(react=True, retract_mm=RETRACT_MM))]
    for lbl, kw in combos:
        s = sweep(seeds=seeds, grip=True, **kw)
        D[lbl] = s
        print(f"{lbl:>30s} | {s['blind']:9.2f} | {s['df_held']:11.3f} | {s['vres']:11.1f} | "
              f"{s['e_min']*1e3:10.4f} | {s['final']:11.1f}")
    print("  수동성(E_min ≥ 0)과 과제 완주가 전부 유지된다 — 추가한 것이 **국소·소산**이거나 술자 쪽")
    print("  규칙이라 채널 장부를 건드리지 않는다. exp 58 이 세운 경계도 그대로다.")
    print("  권장 조합은 **붙들기 + 술자 반응**이다: 맹행을 늘리지 않고 복귀 돌진만 줄인다.")

    # ---------------- E ----------------
    print("-" * 96)
    print("[E] 이 실험이 실제로 남기는 것")
    print("  · **정책을 비교하기 전에 '이 모델이 그 위해를 표현하는가'를 물어야 한다.** A 절에서")
    print("    붙들기와 후퇴가 구분되지 않은 것은 두 정책이 같아서가 아니라 조직 모델에 탄성이")
    print("    없어서였다. 표현력이 없는 모델에서의 '차이 없음'은 결과가 아니라 침묵이다.")
    print("  · **대책이 그 대책이 필요한 이유를 가릴 수 있다.** 마스터 제동은 어긋남을 줄이면서")
    print("    술자가 반응해야 할 바로 그 신호를 없앤다. exp 55 의 점-대-평면(잔차를 낮추면서 정확도를")
    print("    나쁘게)과 같은 계열이다.")
    print("  · 남은 것: 파악 강성·미끄러짐 한계를 **실측**에서(지금은 자릿수만 맞춘 값이고, B 절의")
    print("    결론이 바로 그 값에 달려 있다), 조직 **점탄성**(exp 53 이 남긴 것), 그리고 후퇴를")
    print("    어디까지 할 것인가의 임상 판단(완전 발관 포함).")

    # ---------------- 그림 ----------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    ax = axes[0, 0]
    bb = sorted(A)
    ax.plot(bb, [A[b][0] for b in bb], "-o", color="0.55",
            label="cutting + friction (the chain's model)")
    ax.plot(bb, [A[b][1] for b in bb], "-s", color="crimson",
            label="+ stick-slip grip")
    ax.axhline(2 * F_SLIP, color="0.3", ls="--", lw=1, label="2 × slip limit")
    ax.set_xlabel("patient motion amplitude [mm]")
    ax.set_ylabel("tissue force swing while held [N]")
    ax.set_title("The old model cannot express the hazard at all", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[0, 1]
    lbls = list(B)
    x = np.arange(2)
    ax.bar(x - 0.2, [B[l]["df_held"] for l in lbls], 0.4, color="tab:blue",
           label="extra tissue load [N]")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, [B[l]["blind"] for l in lbls], 0.4, color="crimson",
            label="worst blind travel [mm]")
    ax.set_xticks(x); ax.set_xticklabels(["hold", f"retract {RETRACT_MM:.0f} mm"],
                                         fontsize=9)
    ax.set_ylabel("extra tissue load [N]", color="tab:blue")
    ax2.set_ylabel("blind travel [mm]", color="crimson", fontsize=9)
    ax.set_title("Retraction buys little and costs motion without information",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 2]
    names = ["fixed op\nmaster free", "fixed op\nmaster locked",
             "reacting op\nmaster free", "reacting op\nmaster locked"]
    keys = [("고정 임피던스(exp 50~58)", "자유(exp 58)"),
            ("고정 임피던스(exp 50~58)", "국소 제동"),
            (rk, "자유(exp 58)"), (rk, "국소 제동")]
    xa = np.arange(4)
    ax.bar(xa - 0.2, [C[k]["mism"] for k in keys], 0.4, color="0.6",
           label="mismatch at release [mm]")
    ax2 = ax.twinx()
    ax2.bar(xa + 0.2, [C[k]["vres"] for k in keys], 0.4, color="crimson",
            label="peak speed on resume [mm/s]")
    ax.set_xticks(xa); ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("mismatch [mm]")
    ax2.set_ylabel("resume peak [mm/s]", color="crimson", fontsize=9)
    ax.set_title("The lock hides the cue the operator would react to", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    for grip, c, lbl in ((False, "0.55", "cutting + friction"),
                         (True, "crimson", "+ stick-slip grip")):
        ts = jc.tele.Tissue() if not grip else GrippingTissue()
        x0 = X_SURFACE
        for _ in range(4000):
            x0 += 12.0e-3 / 4000.0
            ts.force(x0)
        tt = np.arange(0, 4.0, DT)
        ff = [abs(ts.force(x0 - BREATH_MM * 1e-3
                           * np.sin(2 * np.pi * BREATH_HZ * s))) for s in tt]
        ax.plot(tt, ff, color=c, lw=1.3, label=lbl)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tissue force [N]")
    ax.set_title(f"Tool held at depth, patient moving {BREATH_MM:.0f} mm", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    r_h = run(seed=0, grip=True, retract_mm=0.0)
    r_r = run(seed=0, grip=True, retract_mm=RETRACT_MM)
    for r, c, lbl in ((r_h, "crimson", "hold"),
                      (r_r, "tab:blue", f"retract {RETRACT_MM:.0f} mm")):
        ax.plot(r["log"]["t"], r["log"]["xs"] * 1e3, color=c, lw=1.3, label=lbl)
    held = r_h["log"]["held"] > 0.5
    ax.fill_between(r_h["log"]["t"], 0, 60, where=held, color="0.5", alpha=0.15,
                    label="stopped")
    ax.axhline(X_TARGET * 1e3, color="0.3", ls="--", lw=1)
    ax.axhline(X_SURFACE * 1e3, color="0.6", ls=":", lw=1)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tool depth [mm]"); ax.set_ylim(0, 60)
    ax.set_title("Hold versus retract on the task", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    ax = axes[1, 2]
    cl = list(D)
    xc = np.arange(len(cl))
    en = ["exp 58 as is", "+ reacting operator", "+ operator + retract"]
    ax.bar(xc - 0.2, [D[l]["blind"] for l in cl], 0.4, color="crimson",
           label="blind travel [mm]")
    ax2 = ax.twinx()
    ax2.bar(xc + 0.2, [D[l]["vres"] for l in cl], 0.4, color="tab:blue",
            label="resume peak [mm/s]")
    ax.set_xticks(xc); ax.set_xticklabels(en, fontsize=7, rotation=15, ha="right")
    ax.set_ylabel("blind travel [mm]", color="crimson")
    ax2.set_ylabel("resume peak [mm/s]", color="tab:blue", fontsize=9)
    ax.set_title("Hold + a reacting operator is the combination", fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("59. Is stopping a safe state? — the question needed a better model "
                 "before a better controller", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "59_what_is_safe_state.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/59_what_is_safe_state.png, assets/59_what_is_safe_state.png")

    return dict(A=A, B=B, C=C, D=D)


# --------------------------------------------------------------------------- #
# 한계·트레이드오프
#   - **B 절의 결론이 파악 파라미터에 직접 달려 있다.** K_GRIP·F_SLIP 은 자릿수만 맞춘 값이고 실측이
#     아니다. 미끄러짐 한계가 크면(조직이 잘 안 놓아준다) 후퇴가 이기는 쪽으로 넘어간다 — 즉 이
#     실험은 "붙들기가 맞다"를 주장하지 않고 **"어느 조직인가가 정책을 고른다"**를 주장한다.
#   - stick-slip 은 축방향 1-DOF 다. 실제로는 축 둘레 파악·횡방향 지지·조직 점탄성(누른 뒤 이완)이
#     같이 작용한다. 점탄성은 exp 53 이 남긴 항목 그대로 미해결이다.
#   - 환자 움직임을 단일 정현파로 뒀다. 호흡은 비대칭이고 심박이 겹치며, 뇌수술처럼 프레임으로
#     고정하면 훨씬 작다 — 진폭 5 mm 는 복부·흉부 계열의 값이다.
#   - 적응형 술자는 "손과 도구의 어긋남이 보이면 반응시간 뒤에 손을 멈춘다"는 **한 규칙**이다.
#     실제 술자는 힘으로도 느끼고, 학습하고, 되돌리기도 한다. exp 50 이 폐기한 '이득 있는 시각
#     폐루프'와 달리 발산하지는 않지만, 이것도 여전히 모델이다.
#   - 후퇴는 조직 안에서만, 표면까지만 한다. **완전 발관**은 별개의 임상 결정이라 넣지 않았다.
#   - 채널·정지·플랜트의 이상화는 exp 56~58 에서 그대로 물려받는다.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
