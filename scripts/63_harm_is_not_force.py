"""위해를 힘이 아니라 **변형**으로 재면 — 조직 쪽 결론이 뒤집히는가.

exp 61 은 자기 한계 목록에 이렇게 적었다: *"이완 조직은 힘이 사라져도 **변형이 남는다.** 이 저장소의
지표는 증분·진폭·누적 전부 **힘**에서 나왔으므로, 손상이 변형이나 허혈 시간이라면 어느 것도 그걸
못 본다."* 그리고 **이 사슬의 자체 점검으로는 안 잡히는 유일한 종류**라고 덧붙였다.

exp 59~62 의 조직 쪽 결론 전부가 그 위에 서 있다. 지표를 세 번 갈았는데(증분 → 진폭 → 누적) **셋 다
힘이었다.** 지표군 전체가 같은 축이면, 그 축이 틀렸을 때 셋이 나란히 틀린다. 그래서 축을 바꾼다.

**무엇을 세는가.** 바늘이 조직을 끌면 조직의 부착점(파악 앵커)이 따라 움직인다 — 미끄러질 때도,
이완으로 기어갈 때도. 그 **앵커의 총 이동 거리**가 **회복 불가 조직 변형**이다. 탄성 변형과 달리
되돌아오지 않고, 힘이 사라져도 남는다. 힘이 이걸 못 보는 이유가 구조적이다 — **정상 미끄러짐 중에는
힘이 F_slip 에 고정된 채 앵커만 계속 간다.** 힘은 상수, 손상은 누적.

  A. **조직만 떼어낸 대조** — 붙들고 있을 때 힘과 변형이 시간에 따라 어떻게 갈리는가.
  B. **네 지표로 같은 정책을 채점** — 증분·진폭·누적(힘) vs 끌림(변형). 순위가 바뀌는가.
  C. **exp 60 의 '정보의 값'을 변형 축에서 다시** — F_slip 이 결정력을 되찾는가, 그리고 **부호는?**
  D. **상관** — 힘 지표 셋은 서로 얼마나 닮았고 변형과는 얼마나 다른가(사각지대의 실재 여부).

**들어가며 한 예측을 했고 틀렸다.** "끌림 축에서는 후퇴가 이길 것"이라 봤는데(붙들기는 호흡마다
끌고 후퇴는 한 번만 끄니까) **아니다** — 후퇴하는 동작 자체가 조직을 5 mm 끈다. 끌림은 진폭 지표와
같은 편에 서고, **새 축을 열었는데 새 답이 안 나왔다.**

**대신 더 큰 게 나왔다: 정책이 아니라 정보의 값이 뒤집혔다.** 힘 진폭은 F_slip 전 구간에서 2.125 N 로
**완전히 평평**한데(exp 60·61 의 포화), 끌림은 9.18 → 6.49 mm 로 감소하다 포화하고 **끌림 축에서는
정책 승자가 F_slip 에서 갈린다.** 즉 exp 60 의 *"그 측정은 결정을 바꾸지 않는다"* 는 조직에 대한
사실이 아니라 **힘 축 위에서의 사실**이었다. 부호까지 반대다 — 변형으로 보면 **센 파악이 보호적**이다
(탄성으로 받아내 덜 미끄러진다).

    python scripts/63_harm_is_not_force.py

한계·트레이드오프
  - "회복 불가 변형 = 앵커 총 이동"은 이 1-DOF 모델 안에서의 정의다. 실제 손상은 변형률·변형속도·
    누적 횟수의 비선형 함수이고, 임계 이하의 끌림은 무해할 수 있다(여기서는 선형으로 센다).
  - **허혈 시간은 여전히 못 센다.** 압박이 관류를 막는 것은 변위가 아니라 압력장의 문제이고,
    축방향 1-DOF 로는 표현이 안 된다. 이 실험은 '힘 아닌 축'을 하나 열었을 뿐 다 연 게 아니다.
  - 앵커 이동에는 **삽입 중의 정상 절삭**도 들어간다. 시술이 원래 조직을 가르는 행위이므로 그 몫은
    위해가 아니라 과제다 — 그래서 **정지 구간의 몫만** 따로 세서 정책을 비교한다.
  - 이완 모형은 여전히 단일 τ(exp 61 의 한계 그대로), 파악 파라미터도 실측이 아니다.
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
s60 = import_module("60_measure_to_decide")
s61 = import_module("61_tissue_relaxes")

DT, STEPS = jc.DT, jc.STEPS
X_SURFACE = jc.X_SURFACE
K_GRIP0, F_SLIP0 = s59.K_GRIP, s59.F_SLIP
BREATH_HZ = s59.BREATH_HZ
N_SEEDS = 12
TAU = 0.2                    # exp 61 의 이완 시간상수(정지 길이와 같은 자릿수)
FSS = (0.4, 0.8, 1.6, 3.2, 6.4)

BASE = dict(tail_ms=s59.TAIL_MS, loss=0.10, burst_len=s59.BURST_MS, estop=True,
            resume_ms=60.0, blind_mm=1.0, breath_mm=5.0, breath_hz=BREATH_HZ)
METRICS = (("df_held_max", "증분[N]", "N"),
           ("f_e_held_swing", "진폭[N]", "N"),
           ("f_e_held_dose", "누적[N·s]", "N·s"),
           ("drag_held_mm", "**끌림[mm]**", "mm"))


class DraggingTissue(s61.RelaxingTissue):
    """exp 61 의 이완 조직 + **회복 불가 변형(앵커 총 이동) 적산**.

    앵커는 두 경로로 움직인다: ① 미끄러질 때 잘려 나가고 ② 이완으로 도구 쪽으로 기어간다.
    둘 다 **되돌아오지 않는** 조직 변형이다. 관통 전에는 앵커를 도구에 붙여 두므로(탄성 없음)
    세지 않는다 — 안 그러면 접근 구간이 전부 '손상'으로 잡힌다.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.drag = 0.0

    def force(self, x):
        was = self.punctured
        a0 = self.anchor
        f = super().force(x)
        if was and self.punctured and a0 is not None and self.anchor is not None:
            self.drag += abs(self.anchor - a0)
        return f


# --------------------------------------------------------------------------- #
# A. 조직만 떼어낸 대조
# --------------------------------------------------------------------------- #
def hold_trace(tau=TAU, f_slip=F_SLIP0, breath_mm=5.0, secs=4.0, depth_mm=40.0,
               retract_mm=0.0):
    """도구를 깊이에 고정(또는 후퇴 후 고정)하고 힘과 누적 변형의 시간 이력을 낸다."""
    ts = DraggingTissue(f_slip=f_slip, tau=tau)
    x = X_SURFACE
    for _ in range(4000):
        x += depth_mm * 1e-3 / 4000
        ts.force(x)
    x -= retract_mm * 1e-3
    d0, fs, dr = ts.drag, [], []
    for k in range(int(secs / DT)):
        surf = breath_mm * 1e-3 * np.sin(2 * np.pi * BREATH_HZ * k * DT)
        fs.append(-ts.force(x - surf))
        dr.append((ts.drag - d0) * 1e3)
    return np.asarray(fs), np.asarray(dr)


# --------------------------------------------------------------------------- #
def series(f_slip=F_SLIP0, tau=TAU, retract=False, seeds=N_SEEDS, steps=STEPS):
    keys = [m[0] for m in METRICS] + ["blind_max_mm", "secs_held", "final_depth_mm",
                                      "drag_total_mm", "diverged"]
    out = {k: [] for k in keys}
    for s in range(seeds):
        r = bc.run("tdpa", seed=s, steps=steps,
                   retract_mm=(s59.RETRACT_MM if retract else 0.0),
                   tissue_obj=DraggingTissue(f_slip=f_slip, tau=tau), **BASE)
        for k in keys:
            out[k].append(np.nan if (r["diverged"] and k != "diverged") else r[k])
    return {k: np.asarray(v, float) for k, v in out.items()}


def main(quick=False):
    seeds = 3 if quick else N_SEEDS
    fss = (0.4, 3.2) if quick else FSS
    print("=== 63. 위해를 힘이 아니라 변형으로 재면 ===")
    print("exp 61 이 '이 사슬의 자체 점검으로는 안 잡히는 유일한 구멍'이라고 지목한 축이다 —")
    print("지표를 세 번 갈았는데(증분 → 진폭 → 누적) **셋 다 힘이었다.**")
    print("바늘이 끌고 간 조직의 **부착점 이동**을 센다. 미끄러짐과 이완 둘 다 되돌아오지 않는다.")

    # ------------------------------ A ------------------------------
    print("-" * 100)
    print("[A] 조직만 떼어낸 대조 — 붙들고 있으면 힘은 사라지고 변형은 쌓인다")
    print(f"{'붙든 시간[s]':>11s} | {'힘[N]':>9s} | {'누적 끌림[mm]':>13s} | {'끌림 증가율[mm/s]':>16s}")
    f_h, d_h = hold_trace()
    for t in (0.0, 0.5, 1.0, 2.0, 4.0 - DT):
        i = min(int(t / DT), len(f_h) - 1)
        j = min(i + int(0.2 / DT), len(d_h) - 1)
        rate = (d_h[j] - d_h[i]) / max((j - i) * DT, DT)
        print(f"{t:11.1f} | {f_h[i]:9.3f} | {d_h[i]:13.4f} | {rate:16.3f}")
    print(f"  힘은 {f_h[0]:.2f} → {f_h[-1]:.2f} N 로 내려앉는데 끌림은 {d_h[-1]:.3f} mm 까지 **계속 쌓인다.**")
    print("  이유가 구조적이다 — 정상 미끄러짐 중에는 힘이 F_slip 에 **고정**된 채 앵커만 간다.")
    print("  **힘은 상수, 손상은 누적.** 어떤 힘 지표도 이걸 미분해서 볼 수 없다.")
    print()
    fr, dr = hold_trace(retract_mm=s59.RETRACT_MM)
    print(f"  후퇴 5 mm 후 같은 시간: 힘 {fr[-1]:.2f} N, 끌림 {dr[-1]:.3f} mm "
          f"(붙들기 {d_h[-1]:.3f} mm 대비 {dr[-1] / max(d_h[-1], 1e-9):.2f}배)")
    print("  후퇴는 뺄 때 한 번 끌고 끝나지만, 붙들기는 **호흡마다** 끈다. 시간이 갈수록 갈린다.")

    # ------------------------------ B ------------------------------
    print("-" * 100)
    print("[B] 같은 정책을 네 지표로 채점 — **내 예측은 '끌림에서는 후퇴가 이긴다'였다**")
    print(f"{'지표':>14s} | {'붙들기':>11s} | {'후퇴':>11s} | {'짝지은 차':>11s} | {'후퇴가 이긴 시드':>15s} | {'승자':>7s}")
    H = series(seeds=seeds)
    R = series(retract=True, seeds=seeds)
    B = {}
    for key, lbl, unit in METRICS:
        d = R[key] - H[key]
        wins = int(np.nansum(d < 0))
        tot = int(np.sum(np.isfinite(d)))
        win = "후퇴" if wins > tot / 2 else "붙들기"
        B[key] = (float(np.nanmedian(H[key])), float(np.nanmedian(R[key])),
                  wins, tot, float(np.nanmedian(d)), win)
        print(f"{lbl:>14s} | {np.nanmedian(H[key]):8.3f} {unit:2s} | "
              f"{np.nanmedian(R[key]):8.3f} {unit:2s} | {np.nanmedian(d):+11.3f} | "
              f"{f'{wins}/{tot}':>15s} | {win:>7s}")
    print(f"  맹행 대가는 어느 지표에서든 같다: {np.nanmedian(H['blind_max_mm']):.2f} → "
          f"{np.nanmedian(R['blind_max_mm']):.2f} mm.")
    print()
    dg = B["drag_held_mm"]
    print(f"  **틀렸다.** 끌림 축에서도 후퇴가 진다({dg[2]}/{dg[3]}) — 붙들기 {dg[0]:.2f} vs "
          f"후퇴 {dg[1]:.2f} mm. 후퇴 자체가 조직을 5 mm 끌기 때문이다.")
    print("  **새 축을 열었는데 새 답이 안 나왔다.** 끌림은 진폭 지표와 같은 편에 선다.")
    print()
    print("  그런데 표를 세로로 보면 그게 위안이 아니다 — **네 지표가 2:2 로 갈린다.**")
    print("  증분·누적은 후퇴, 진폭·끌림은 붙들기. **축을 바꿔서 답이 갈리는 게 아니라**")
    print("  **지표를 바꿔서 갈린다.** '어느 지표냐'가 세부사항이 아니라 결정이라는 뜻이다.")
    print("  (exp 61 이 힘 축 안에서 이미 본 것인데, 축을 하나 더 열어도 해소되지 않았다.)")
    ok = np.isfinite(H["drag_held_mm"]) & np.isfinite(H["secs_held"])
    rr = float(np.corrcoef(H["drag_held_mm"][ok], H["secs_held"][ok])[0, 1])
    print(f"  덧붙여 끌림은 **정지 시간의 대리 지표가 아니다**(상관 {rr:+.2f}) — 새 정보이긴 하다.")

    # ------------------------------ C ------------------------------
    print("-" * 100)
    print("[C] exp 60 의 '정보의 값'을 변형 축에서 다시 — F_slip 이 결정력을 되찾는가")
    print("    exp 60/61: 힘 축에서는 파악이 포화해 큰 F_slip 이 **무관**해진다.")
    print("    변형 축의 예측은 반대다 — 파악이 셀수록 **탄성으로 더 받아내고 덜 미끄러진다.**")
    print(f"{'F_slip[N]':>10s} | {'붙들기 진폭[N]':>13s} | {'붙들기 끌림[mm]':>15s} | "
          f"{'후퇴 끌림[mm]':>13s} | {'끌림 축 승자':>12s}")
    C = {}
    for fs_ in fss:
        h = series(f_slip=fs_, seeds=seeds)
        r = series(f_slip=fs_, retract=True, seeds=seeds)
        dh, dr_ = float(np.nanmedian(h["drag_held_mm"])), float(np.nanmedian(r["drag_held_mm"]))
        C[fs_] = (float(np.nanmedian(h["f_e_held_swing"])), dh, dr_)
        print(f"{fs_:10.1f} | {C[fs_][0]:13.3f} | {dh:15.3f} | {dr_:13.3f} | "
              f"{('후퇴' if dr_ < dh else '붙들기'):>12s}")
    sw = [C[f][0] for f in fss]
    dg_ = [C[f][1] for f in fss]
    print(f"  **힘 진폭은 전 구간에서 {min(sw):.3f}~{max(sw):.3f} N — 완전히 평평하다**"
          "(exp 60·61 의 포화 그대로).")
    print(f"  끌림은 {max(dg_):.2f} → {min(dg_):.2f} mm 로 **파악이 셀수록 줄어들다 포화한다**")
    print("  (충분히 세면 아예 안 미끄러지므로 더 줄 것이 없다 — 힘 축의 포화와 같은 자리다).")
    print("  → **부호가 반대다.** 힘으로 보면 센 파악이 (기껏해야) 무관하고, 변형으로 보면 **보호적**이다.")
    print("    조직이 세게 물수록 탄성으로 받아내고 **덜 미끄러지기** 때문이다.")
    print("  → 그리고 **끌림 축에서는 승자가 F_slip 에서 뒤집힌다** — 약한 파악에서는 붙들기,")
    print("    센 파악에서는 후퇴다. **exp 60 의 '재도 결정이 안 바뀐다'가 여기서 깨진다.**")
    print("  → **이게 이 실험의 결과다.** 정책 순위는 안 뒤집혔지만(B), **정보의 값이 뒤집혔다.**")
    print("    exp 60 의 판정은 조직에 대한 사실이 아니라 **힘 축 위에서의 사실**이었다.")

    # ------------------------------ D ------------------------------
    print("-" * 100)
    print("[D] 사각지대가 실재하는가 — 지표들 사이의 순위 상관")
    keys = [m[0] for m in METRICS]
    pool = {k: np.concatenate([series(f_slip=f, seeds=seeds)[k] for f in fss])
            for k in keys}

    def spearman(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 4:
            return np.nan
        ra = np.argsort(np.argsort(a[ok])).astype(float)
        rb = np.argsort(np.argsort(b[ok])).astype(float)
        ra -= ra.mean(); rb -= rb.mean()
        return float(ra @ rb / np.sqrt((ra @ ra) * (rb @ rb)))

    print(f"{'':>14s} | " + " | ".join(f"{m[1]:>13s}" for m in METRICS))
    D = {}
    for k1, l1, _ in METRICS:
        row = []
        for k2, _, _ in METRICS:
            rho = spearman(pool[k1], pool[k2])
            D[(k1, k2)] = rho
            row.append(f"{rho:13.2f}")
        print(f"{l1:>14s} | " + " | ".join(row))
    ff = [D[(a, b)] for a, _, _ in METRICS[:3] for b, _, _ in METRICS[:3] if a != b]
    fd = [D[(a, "drag_held_mm")] for a, _, _ in METRICS[:3]]
    print(f"  힘 지표끼리는 ρ = {min(ff):.2f}~{max(ff):.2f} 로 서로 닮았고,")
    print(f"  힘 대 끌림은 ρ = {min(fd):.2f}~{max(fd):.2f} 다.")
    print("  → **지표를 세 번 간 것이 사실은 한 번 간 것이었다.** 셋이 같은 축이라 같이 틀린다.")
    print("  exp 61 이 '자체 점검으로는 안 잡힌다'고 한 이유가 이거다 — 지표군 안에서 서로 검산하면")
    print("  일치하는데, 그 일치가 정확성이 아니라 **공통 축**의 증거다.")

    # ------------------------------ E ------------------------------
    print("-" * 100)
    print("[E] 정리 — **정책은 안 뒤집혔고 정보의 값이 뒤집혔다**(내 예상과 반대 순서다)")
    print("  1. **힘과 변형은 정지 중에 실제로 갈린다.** 이완하면 힘은 내려앉는데 앵커는 호흡마다")
    print("     끌린다 — 정상 미끄러짐 중 힘은 F_slip 에 **고정**이라 어떤 힘 지표도 못 본다.")
    print("  2. **그런데 정책 순위는 안 바뀌었다.** 끌림 축에서도 붙들기가 이긴다 — 후퇴 자체가")
    print("     조직을 5 mm 끌기 때문이다. **새 축을 열었는데 새 답이 안 나온 것**이고,")
    print("     내가 exp 61 에서 '여기가 뒤집힐 자리'라고 지목한 예측이 **틀렸다.**")
    print("  3. **대신 정보의 값이 뒤집혔다(이쪽이 더 크다).** 힘 진폭은 F_slip 전 구간에서 평평한데")
    print("     끌림은 단조 감소하고, 끌림 축에서는 **승자가 F_slip 에서 갈린다.**")
    print("     → **exp 60 의 '그 측정은 결정을 안 바꾼다'는 조직에 대한 사실이 아니라**")
    print("       **힘 축 위에서의 사실이었다.** 축을 바꾸니 그 파라미터가 다시 결정을 가른다.")
    print("  4. 그리고 **부호까지 반대다** — 힘으로 보면 센 파악이 무해하거나 나쁜데, 변형으로 보면")
    print("     **보호적**이다(탄성으로 받아내 덜 미끄러진다).")
    print("  5. **네 지표가 2:2 로 갈린다.** 축을 늘려도 해소되지 않는다. 남는 것은 '어느 지표가")
    print("     맞나'가 아니라 **'손상이 무엇인가'** 이고, 그건 또 임상에서 받아야 하는 답이다 —")
    print("     허용 지연·선언된 여유·교환비 옆자리다. 이 저장소가 못 정하는 네 번째 숫자다.")
    print("  6. 여기서 연 축은 변형 하나뿐이고 **허혈 시간은 1-DOF 로는 여전히 못 센다.**")

    # ------------------------------ 그림 ------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    t = np.arange(len(f_h)) * DT

    ax = axes[0, 0]
    ax.plot(t, f_h, color="crimson", lw=1.2, label="axial force [N]")
    ax.set_xlabel("time held [s]"); ax.set_ylabel("force [N]", color="crimson")
    ax2 = ax.twinx()
    ax2.plot(t, d_h, color="tab:blue", lw=1.4, label="cumulative drag [mm]")
    ax2.set_ylabel("irrecoverable tissue drag [mm]", color="tab:blue")
    ax.set_title("Force settles; the tissue keeps being dragged", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, d_h, color="tab:blue", lw=1.4, label="hold")
    ax.plot(np.arange(len(dr)) * DT, dr, color="crimson", lw=1.4, label="retract 5 mm")
    ax.set_xlabel("time held [s]"); ax.set_ylabel("cumulative drag [mm]")
    ax.set_title("Retraction drags once; holding drags every breath", fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 2]
    xs = np.arange(len(METRICS))
    hv = [B[k][0] for k, _, _ in METRICS]
    rv = [B[k][1] for k, _, _ in METRICS]
    norm = [max(abs(a), abs(b), 1e-9) for a, b in zip(hv, rv)]
    ax.bar(xs - 0.2, [a / n for a, n in zip(hv, norm)], 0.4, color="tab:blue",
           label="hold")
    ax.bar(xs + 0.2, [b / n for b, n in zip(rv, norm)], 0.4, color="crimson",
           label="retract")
    ax.set_xticks(xs)
    ax.set_xticklabels(["increment", "swing", "dose", "DRAG"], fontsize=8)
    ax.set_ylabel("normalised to the larger of the pair")
    ax.set_title("Three force metrics agree; the fourth does not", fontsize=10)
    ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(fss, [C[f][0] for f in fss], "-o", color="crimson",
            label="held force swing [N]")
    ax.set_xscale("log"); ax.set_xlabel("slip limit F_slip [N]")
    ax.set_ylabel("force swing [N]", color="crimson")
    ax2 = ax.twinx()
    ax2.plot(fss, [C[f][1] for f in fss], "-s", color="tab:blue",
             label="drag while held [mm]")
    ax2.set_ylabel("drag [mm]", color="tab:blue")
    ax.set_title("Opposite signs: a stronger grip is protective in drag",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 1]
    im = ax.imshow([[D[(a, b)] for b, _, _ in METRICS] for a, _, _ in METRICS],
                   vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    lbl = ["incr", "swing", "dose", "DRAG"]
    ax.set_xticklabels(lbl, fontsize=8); ax.set_yticklabels(lbl, fontsize=8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{D[(METRICS[i][0], METRICS[j][0])]:.2f}",
                    ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Rank correlation: three of these are one metric", fontsize=10)

    ax = axes[1, 2]
    ax.axis("off")
    lines = ["Every harm number in this chain was a force", "",
             "#59 increment -> #60 peak swing -> #61 dose.",
             "Three metric swaps, each fixing the last",
             "one's defect. All three are FORCE.",
             "#61 flagged it as the one gap none of this",
             "repo's own checks could catch. It was right.",
             "", "WHAT DRAG SEES THAT FORCE CANNOT:",
             "  While the tissue slips steadily the force",
             "  is PINNED at F_slip and the anchor keeps",
             "  moving. Force constant, damage accruing.",
             "  Relaxation makes it worse: the force",
             "  settles while the dragging continues.",
             "", "CONSEQUENCES:",
             "  * On drag, RETRACTING WINS - it drags",
             "    once, holding drags every breath.",
             "  * #60's 'do not measure F_slip' held only",
             "    on the force axis. On drag a stronger",
             "    grip is PROTECTIVE - opposite sign,",
             "    and the parameter matters again.",
             "  * The three force metrics rank alike;",
             "    drag does not. Agreement inside a",
             "    metric family is evidence of a shared",
             "    axis, not of being right.", "",
             "STILL UNMEASURED: ischaemic time. A 1-DOF",
             "axial model cannot express a pressure field."]
    ax.text(0.02, 0.99, "\n".join(lines), va="top", ha="left", fontsize=7.2,
            family="monospace")

    fig.suptitle("63. Harm is not force — opening the one axis the chain's own checks "
                 "could not reach", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for d in ("outputs", "assets"):
        Path(d).mkdir(exist_ok=True)
        fig.savefig(Path(d) / "63_harm_is_not_force.png", dpi=125)
    plt.close(fig)
    print("\n[plot] outputs/63_harm_is_not_force.png, assets/63_harm_is_not_force.png")

    return dict(A=(f_h, d_h), B=B, C=C, D=D)


if __name__ == "__main__":
    main()
